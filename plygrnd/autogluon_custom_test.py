import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular import TabularPredictor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class CustomSVMModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')

        from sklearn.svm import SVC, SVR

        if self.problem_type in ['regression', 'softclass']:
            model_cls = SVR
        else:
            model_cls = SVC

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'probability': True,
            'tol': 1e-3,
            'cache_size': 200,
            'class_weight': None,
            'verbose': False,
            'max_iter': -1,
            'decision_function_shape': 'ovr',
            'break_ties': False,
            'random_state': None,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=['int', 'float', 'category'],
            ),
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


# 加载Iris数据集
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['label'] = iris.target

# 将数据集拆分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
label = 'label'


# 创建一个TabularPredictor实例
predictor = TabularPredictor(
    label=label,
    problem_type='multiclass',
    eval_metric='accuracy',
)

# 设置自定义模型的超参数
custom_hyperparameters = {CustomSVMModel: {}}
# custom_hyperparameters = {'RF': {}}

# 使用自定义SVM模型进行训练
predictor.fit(
    train_data,
    hyperparameters=custom_hyperparameters,
)

# 对测试集进行预测
y_pred = predictor.predict(test_data)

# 评估模型性能
score = predictor.evaluate(test_data)
print('Test accuracy:', score)
