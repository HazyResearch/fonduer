import logging
from multiprocessing import Manager, Process
from queue import Empty, Queue
from typing import Any, Collection, Dict, Iterator, List, Optional, Set, Type

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from fonduer.meta import Meta, new_sessionmaker
from fonduer.parser.models.document import Document

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm


class UDFRunner(object):
    """
    Class to run UDFs in parallel using simple queue-based multiprocessing
    setup.
    """

    def __init__(
        self,
        session: Session,
        udf_class: Type["UDF"],
        parallelism: int = 1,
        **udf_init_kwargs: Any,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.udf_class = udf_class
        self.udf_init_kwargs = udf_init_kwargs
        self.udfs: List["UDF"] = []
        self.pb = None
        self.session = session
        self.parallelism = parallelism

        #: The last set of documents that apply() was called on
        self.last_docs: Set[str] = set()

    def apply(
        self,
        doc_loader: Collection[
            Document
        ],  # doc_loader has __len__, but Iterable doesn't.
        clear: bool = True,
        parallelism: Optional[int] = None,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Apply the given UDF to the set of objects returned by the doc_loader, either
        single or multi-threaded, and optionally calling clear() first.
        """
        # Clear everything downstream of this UDF if requested
        if clear:
            self.clear(**kwargs)

        # Track the last documents parsed by apply
        self.last_docs = set(doc.name for doc in doc_loader)

        # Execute the UDF
        self.logger.info("Running UDF...")

        # Setup progress bar
        if progress_bar:
            self.logger.debug("Setting up progress bar...")
            if hasattr(doc_loader, "__len__"):
                self.pb = tqdm(total=len(doc_loader))
            else:
                self.logger.error("Could not determine size of progress bar")

        # Use the parallelism of the class if none is provided to apply
        parallelism = parallelism if parallelism else self.parallelism
        self._apply(doc_loader, parallelism, clear=clear, **kwargs)

        # Close progress bar
        if self.pb is not None:
            self.logger.debug("Closing progress bar...")
            self.pb.close()

        self.logger.debug("Running after_apply...")
        self._after_apply(**kwargs)

    def clear(self, **kwargs: Any) -> None:
        """Clear the associated data from the database."""
        raise NotImplementedError()

    def _after_apply(self, **kwargs: Any) -> None:
        """This method is executed by a single process after apply."""
        pass

    def _add(self, instance: Any) -> None:
        pass

    def _apply(
        self, doc_loader: Collection[Document], parallelism: int, **kwargs: Any
    ) -> None:
        """Run the UDF multi-threaded using python multiprocessing"""
        if not Meta.postgres:
            raise ValueError("Fonduer must use PostgreSQL as a database backend.")

        # Create an input queue to feed documents to UDF workers
        manager = Manager()
        in_queue = manager.Queue()
        # Use an output queue to track multiprocess progress
        out_queue = manager.Queue()

        # Fill input queue with documents
        for doc in doc_loader:
            in_queue.put(doc)
        total_count = in_queue.qsize()

        # Create UDF Processes
        for i in range(parallelism):
            udf = self.udf_class(
                in_queue=in_queue,
                out_queue=out_queue,
                worker_id=i,
                **self.udf_init_kwargs,
            )
            udf.apply_kwargs = kwargs
            self.udfs.append(udf)

        # Start the UDF processes
        for udf in self.udfs:
            udf.start()

        count_parsed = 0
        while (
            any([udf.is_alive() for udf in self.udfs]) or not out_queue.empty()
        ) and count_parsed < total_count:
            try:
                y = out_queue.get(timeout=1)
                self._add(y)
                # Update progress bar whenever an item has been processed
                count_parsed += 1
                if self.pb is not None:
                    self.pb.update(1)
            except Empty:
                # This happens when any child process is alive and still processing.
                pass
            except Exception as e:
                # Raise an error for all the other exceptions.
                raise (e)

        # Join the UDF processes
        for udf in self.udfs:
            udf.join()

        # Flush the processes
        self.udfs = []

        self.session.commit()


class UDF(Process):
    TASK_DONE = "done"

    def __init__(
        self,
        in_queue: Optional[Queue] = None,
        out_queue: Optional[Queue] = None,
        worker_id: int = 0,
        **udf_init_kwargs: Any,
    ) -> None:
        """
        :param in_queue: A Queue of input objects to processes
        :param out_queue: A Queue of output objects from processes
        :param worker_id: An ID of a process
        """
        super().__init__()
        self.daemon = True
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.worker_id = worker_id

        # We use a workaround to pass in the apply kwargs
        self.apply_kwargs: Dict[str, Any] = {}

    def run(self) -> None:
        """
        This method is called when the UDF is run as a Process in a
        multiprocess setting The basic routine is: get from JoinableQueue,
        apply, put / add outputs, loop
        """
        # Each UDF starts its own Engine
        # See SQLalchemy, using connection pools with multiprocessing.
        Session = new_sessionmaker()
        session = Session()
        while True:
            try:
                doc = self.in_queue.get_nowait()
                # Merge the object with the session owned by the current child process.
                # If transient (ie not saved), save the object to the database.
                # If not, load it from the database w/o the overhead of reconciliation.
                if inspect(doc).transient:  # This only happens during parser.apply
                    doc = session.merge(doc, load=True)
                else:
                    doc = session.merge(doc, load=False)
                y = self.apply(doc, **self.apply_kwargs)
                self.out_queue.put(y)
            except Empty:
                break
        session.commit()
        session.close()

    def apply(self, doc: Document, **kwargs: Any) -> Iterator[Meta.Base]:
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()
