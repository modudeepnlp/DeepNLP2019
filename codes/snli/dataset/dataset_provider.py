import sys
sys.path.append('../')

from configs.model_config import Config

class DatasetProvider:

    def __init__(self, config, experiment_dir=None):

        self.config = config

        if experiment_dir is not None:
            self.config = Config(config.domain_id, experiment_dir)

        self._train_input_fn = None
        self._validate_input_fn = None
        self._test_input_fn = None
        self._train_hook = None
        self._validate_hook = None

    def setup_train_input_graph(self):
        raise NotImplementedError

    def setup_validate_input_graph(self):
        raise NotImplementedError

    def setup_test_input_graph(self, input_type, input_value):
        raise NotImplementedError

    @property
    def train_input_fn(self):
        """
        train data input_fn
        call by estimator.train() in task runner e.g., ner_runner
        :return:
        """
        if self._train_input_fn is None:
            self.setup_train_input_graph()
        return self._train_input_fn

    @property
    def validate_input_fn(self):
        if self._validate_input_fn is None:
            self.setup_validate_input_graph()
        return self._validate_input_fn

    @property
    def test_input_fn(self):
        return self._test_input_fn

    @property
    def train_hook(self):
        if self._train_hook is None:
            self.setup_train_input_graph()
        return self._train_hook

    @property
    def validate_hook(self):
        if self._validate_hook is None:
            self.setup_validate_input_graph()
        return self._validate_hook
