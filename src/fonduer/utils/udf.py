import logging
from multiprocessing import JoinableQueue, Manager, Process
from queue import Empty

from fonduer.meta import Meta, new_sessionmaker

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


QUEUE_TIMEOUT = 3

# Grab pointer to global metadata
_meta = Meta.init()


class UDFRunner(object):
    """
    Class to run UDFs in parallel using simple queue-based multiprocessing
    setup.
    """

    def __init__(self, session, udf_class, parallelism=1, **udf_init_kwargs):
        self.logger = logging.getLogger(__name__)
        self.udf_class = udf_class
        self.udf_init_kwargs = udf_init_kwargs
        self.udfs = []
        self.pb = None
        self.session = session
        self.parallelism = parallelism

        #: The last set of documents that apply() was called on
        self.last_docs = set()

    def apply(
        self, doc_loader, clear=True, parallelism=None, progress_bar=True, **kwargs
    ):
        """
        Apply the given UDF to the set of objects returned by the doc_loader, either
        single or multi-threaded, and optionally calling clear() first.
        """
        # Clear everything downstream of this UDF if requested
        if clear:
            self.clear(**kwargs)

        # Clear the last operated documents
        self.last_docs.clear()

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
        if parallelism < 2:
            self._apply_st(doc_loader, clear=clear, **kwargs)
        else:
            self._apply_mt(doc_loader, parallelism, clear=clear, **kwargs)

        # Close progress bar
        if self.pb is not None:
            self.logger.debug("Closing progress bar...")
            self.pb.close()

    def clear(self, **kwargs):
        """Clear the associated data from the database."""
        raise NotImplementedError()

    def get_last_documents(self):
        """Return the last set of documents that was operated on with apply().

        :rtype: list of ``Documents`` operated on in the last call to ``apply()``.
        """
        return list(self.last_docs)

    def _apply_st(self, doc_loader, **kwargs):
        """Run the UDF single-threaded, optionally with progress bar"""
        udf = self.udf_class(**self.udf_init_kwargs)

        # Run single-thread
        for doc in doc_loader:
            self.last_docs.add(doc)
            if self.pb is not None:
                self.pb.update(1)

            udf.session.add_all(y for y in udf.apply(doc, **kwargs))

        # Commit session and close progress bar if applicable
        udf.session.commit()

    def _apply_mt(self, doc_loader, parallelism, **kwargs):
        """Run the UDF multi-threaded using python multiprocessing"""
        if not _meta.postgres:
            raise ValueError("Fonduer must use PostgreSQL as a database backend.")

        def fill_input_queue(in_queue, doc_loader, terminal_signal):
            for doc in doc_loader:
                in_queue.put(doc)
            in_queue.put(terminal_signal)

        # Create an input queue to feed documents to UDF workers
        manager = Manager()
        in_queue = manager.Queue()
        # Use an output queue to track multiprocess progress
        out_queue = JoinableQueue()

        total_count = len(doc_loader)

        for doc in doc_loader:
            self.last_docs.add(doc)

        # Start UDF Processes
        for i in range(parallelism):
            udf = self.udf_class(
                in_queue=in_queue,
                out_queue=out_queue,
                worker_id=i,
                **self.udf_init_kwargs
            )
            udf.apply_kwargs = kwargs
            self.udfs.append(udf)

        # Start the UDF processes, and then join on their completion
        for udf in self.udfs:
            udf.start()

        # Fill input queue with documents
        terminal_signal = UDF.QUEUE_CLOSED
        in_queue_filler = Process(
            target=fill_input_queue, args=(in_queue, doc_loader, terminal_signal)
        )
        in_queue_filler.start()

        count_parsed = 0
        while count_parsed < total_count:
            y = out_queue.get()
            # Update progress bar whenever an item has been  processed
            if y == UDF.TASK_DONE:
                count_parsed += 1
                if self.pb is not None:
                    self.pb.update(1)
            else:
                raise ValueError("Got non-sentinal output.")

        in_queue_filler.join()
        in_queue.put(UDF.QUEUE_CLOSED)

        for udf in self.udfs:
            udf.join()

        # Terminate and flush the processes
        for udf in self.udfs:
            udf.terminate()
        self.udfs = []


class UDF(Process):
    TASK_DONE = "done"
    QUEUE_CLOSED = "QUEUECLOSED"

    def __init__(self, in_queue=None, out_queue=None, worker_id=0):
        """
        in_queue: A Queue of input objects to process; primarily for running in parallel
        """
        Process.__init__(self)
        self.daemon = True
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.worker_id = worker_id

        # Each UDF starts its own Engine
        # See SQLalchemy, using connection pools with multiprocessing.
        Session = new_sessionmaker()
        self.session = Session()

        # We use a workaround to pass in the apply kwargs
        self.apply_kwargs = {}

    def run(self):
        """
        This method is called when the UDF is run as a Process in a
        multiprocess setting The basic routine is: get from JoinableQueue,
        apply, put / add outputs, loop
        """
        while True:
            try:
                doc = self.in_queue.get(True, QUEUE_TIMEOUT)
                if doc == UDF.QUEUE_CLOSED:
                    self.in_queue.put(UDF.QUEUE_CLOSED)
                    break
                self.session.add_all(y for y in self.apply(doc, **self.apply_kwargs))
                self.out_queue.put(UDF.TASK_DONE)
            except Empty:
                continue
        self.session.commit()
        self.session.close()

    def apply(self, doc, **kwargs):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()
