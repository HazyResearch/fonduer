"""Fonduer UDF."""
import logging
from multiprocessing import Manager, Process
from queue import Queue
from threading import Thread
from typing import Any, Collection, Dict, List, Optional, Set, Type, Union

from sqlalchemy import inspect
from sqlalchemy.orm import Session, scoped_session, sessionmaker

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

logger = logging.getLogger(__name__)


class UDFRunner(object):
    """Class to run UDFs in parallel using simple queue-based multiprocessing setup."""

    def __init__(
        self,
        session: Session,
        udf_class: Type["UDF"],
        parallelism: int = 1,
        **udf_init_kwargs: Any,
    ) -> None:
        """Initialize UDFRunner."""
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
        """Apply the given UDF to the set of objects returned by the doc_loader.

        Either single or multi-threaded, and optionally calling clear() first.
        """
        # Clear everything downstream of this UDF if requested
        if clear:
            self.clear(**kwargs)

        # Execute the UDF
        logger.info("Running UDF...")

        # Setup progress bar
        if progress_bar:
            logger.debug("Setting up progress bar...")
            if hasattr(doc_loader, "__len__"):
                self.pb = tqdm(total=len(doc_loader))
            else:
                logger.error("Could not determine size of progress bar")

        # Use the parallelism of the class if none is provided to apply
        parallelism = parallelism if parallelism else self.parallelism
        self._apply(doc_loader, parallelism, clear=clear, **kwargs)

        # Close progress bar
        if self.pb is not None:
            logger.debug("Closing progress bar...")
            self.pb.close()

        logger.debug("Running after_apply...")
        self._after_apply(**kwargs)

    def clear(self, **kwargs: Any) -> None:
        """Clear the associated data from the database."""
        raise NotImplementedError()

    def _after_apply(self, **kwargs: Any) -> None:
        """Execute this method by a single process after apply."""
        pass

    def _add(self, session: Session, instance: Any) -> None:
        pass

    def _apply(
        self, doc_loader: Collection[Document], parallelism: int, **kwargs: Any
    ) -> None:
        """Run the UDF multi-threaded using python multiprocessing."""
        if not Meta.postgres:
            raise ValueError("Fonduer must use PostgreSQL as a database backend.")

        # Create an input queue to feed documents to UDF workers
        manager = Manager()
        # Set maxsize (#435). The number is heuristically determined.
        in_queue = manager.Queue(maxsize=parallelism * 2)
        # Use an output queue to track multiprocess progress
        out_queue = manager.Queue()

        # Clear the last documents parsed by the last run
        self.last_docs = set()

        # Create DB session factory for insert data on each UDF (#545)
        session_factory = new_sessionmaker()
        # Create UDF Processes
        for i in range(parallelism):
            udf = self.udf_class(
                session_factory=session_factory,
                runner=self,
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

        # Fill input queue with documents but # of docs in queue is capped (#435).
        def in_thread_func() -> None:
            # Do not use session here to prevent concurrent use (#482).
            for doc in doc_loader:
                in_queue.put(doc)  # block until a free slot is available

        Thread(target=in_thread_func).start()

        count_parsed = 0
        total_count = len(doc_loader)

        while (
            any([udf.is_alive() for udf in self.udfs]) or not out_queue.empty()
        ) and count_parsed < total_count:
            # Get doc from the out_queue and persist the result into postgres
            try:
                doc_name = out_queue.get()  # block until an item is available
                self.last_docs.add(doc_name)
                # Update progress bar whenever an item has been processed
                count_parsed += 1
                if self.pb is not None:
                    self.pb.update(1)
            except Exception as e:
                # Raise an error for all the other exceptions.
                raise (e)

        # Join the UDF processes
        for _ in self.udfs:
            in_queue.put(UDF.TASK_DONE)
        for udf in self.udfs:
            udf.join()

        # Flush the processes
        self.udfs = []


class UDF(Process):
    """UDF class."""

    TASK_DONE = "done"

    def __init__(
        self,
        session_factory: sessionmaker = None,
        runner: UDFRunner = None,
        in_queue: Optional[Queue] = None,
        out_queue: Optional[Queue] = None,
        worker_id: int = 0,
        **udf_init_kwargs: Any,
    ) -> None:
        """Initialize UDF.

        :param in_queue: A Queue of input objects to processes
        :param out_queue: A Queue of output objects from processes
        :param worker_id: An ID of a process
        """
        super().__init__()
        self.daemon = True
        self.session_factory = session_factory
        self.runner = runner
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.worker_id = worker_id

        # We use a workaround to pass in the apply kwargs
        self.apply_kwargs: Dict[str, Any] = {}

    def run(self) -> None:
        """Run function of UDF.

        Call this method when the UDF is run as a Process in a
        multiprocess setting The basic routine is: get from JoinableQueue,
        apply, put / add outputs, loop
        """
        # Each UDF get thread local (scoped) session from connection pools
        # See SQLalchemy, using scoped sesion with multiprocessing.
        Session = scoped_session(self.session_factory)
        session = Session()
        while True:
            doc = self.in_queue.get()  # block until an item is available
            if doc == UDF.TASK_DONE:
                break
            # Merge the object with the session owned by the current child process.
            # This does not happen during parsing when doc is transient.
            if not inspect(doc).transient:
                doc = session.merge(doc, load=False)
            y = self.apply(doc, **self.apply_kwargs)
            self.runner._add(session, y)
            self.out_queue.put(doc.name)
        session.commit()
        session.close()
        Session.remove()

    def apply(
        self, doc: Document, **kwargs: Any
    ) -> Union[Document, None, List[List[Dict[str, Any]]]]:
        """Apply function.

        This function takes in an object, and returns a generator / set / list.
        """
        raise NotImplementedError()
