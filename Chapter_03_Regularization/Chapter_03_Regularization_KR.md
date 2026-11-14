# 03. Regularization

## 03.00 Why Regularization

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

정규화(Regularization)는 모델(Model)이 학습 데이터(Training Data)를 단순히 암기하는 대신 의미 있는 패턴(Pattern)을 학습하도록 돕는 기계학습(Machine Learning)의 핵심 원리이다. 최적화 알고리즘(Optimization Algorithm)은 손실 함수(Loss Function)를 최소화하지만, 낮은 학습 손실(Training Loss)이 반드시 새로운 데이터에 대한 신뢰할 수 있는 예측을 보장하지는 않는다. 정규화는 학습 과정(Learning Process)을 더 나은 일반화(Generalization)가 가능한 해(Solution)로 유도하여 이러한 한계를 해결한다.

정규화가 필요한 근본적인 이유는 관측 데이터(Observed Data)에 적합하는 것과 실제 데이터 생성 관계(Data-Generating Relationship)를 학습하는 것 사이에 차이가 있기 때문이다. 충분한 표현 능력(Capacity)을 가진 모델은 무작위 변동(Random Fluctuation)과 측정 오차(Measurement Error)를 포함하여 학습 데이터를 거의 완벽하게 재현할 수 있다. 그러나 이러한 세부 사항이 안정적인 패턴을 의미하지는 않으므로 새로운 데이터에서 예측 정확도(Prediction Accuracy)가 저하될 수 있다.

과적합(Overfitting)은 모델이 학습 데이터셋(Training Dataset)의 특성에 지나치게 민감해질 때 발생한다. 이러한 문제는 조정 가능한 매개변수(Parameter)의 수가 사용 가능한 정보량에 비해 많을 때 특히 빈번하게 나타난다. 심층 신경망(Deep Neural Network)은 수백만 또는 수십억 개의 매개변수를 포함할 수 있으며, 유용한 관계뿐 아니라 불필요한 변동까지 학습할 수 있을 정도로 복잡한 함수(Function)를 표현할 수 있다.

정규화는 학습 알고리즘(Learning Algorithm)이 선택하는 해에 영향을 주는 추가적인 제약 조건(Constraint), 벌점(Penalty), 학습 메커니즘(Training Mechanism)을 도입한다. 경험적 학습 오차(Empirical Training Error)를 최소화하는 데만 집중하는 대신 예측 정확도와 바람직한 구조적·행동적 특성 사이의 균형을 추구한다. 이러한 특성에는 작은 매개변수 크기(Parameter Magnitude), 부드러운 예측(Smooth Prediction), 강건성(Robustness), 잡음(Noise)에 대한 민감도 감소 등이 포함된다.

경험 위험(Empirical Risk)과 기대 위험(Expected Risk)의 차이는 정규화를 이해하기 위한 수학적 기초를 제공한다. 경험 위험은 사용 가능한 학습 샘플(Training Sample)에 대한 평균 예측 오차를 측정하는 반면, 기대 위험은 실제 데이터 분포(Data Distribution) 전체에서의 성능을 나타낸다. 실제 분포는 일반적으로 알려져 있지 않으므로 경험 위험만 최소화하면 관측되지 않은 데이터에서 성능이 저하되는 해가 만들어질 수 있다.

정규화된 최적화 목적 함수(Regularized Optimization Objective)는 일반적으로 J(θ) = L(θ) + λΩ(θ)로 표현된다. 여기서 L(θ)는 경험적 손실(Empirical Loss), Ω(θ)는 정규화 벌점(Regularization Penalty), λ는 벌점의 상대적 영향을 조절하는 정규화 계수(Regularization Coefficient)를 의미한다. 매개변수 벡터(Parameter Vector) θ는 학습 가능한 모델 매개변수를 포함하며, λ가 증가하면 벌점의 영향이 강화되지만 실제 동작은 정규화 방법, 모델 구조, 최적화 과정에 따라 달라진다.

정규화 계수는 학습 데이터에 대한 적합도(Fitting Accuracy)와 학습된 해에 대한 제약 사이의 절충 관계(Tradeoff)를 형성한다. 정규화가 지나치게 약하면 모델이 불필요한 복잡성(Complexity)을 유지하면서 심각한 과적합을 나타낼 수 있다. 반대로 정규화가 지나치게 강하면 중요한 관계까지 억제되어 과소적합(Underfitting)이 발생할 수 있다. 따라서 적절한 정규화는 단순히 벌점을 최대화하는 것이 아니라 서로 경쟁하는 목적 사이의 균형을 찾는 과정이다.

모델 복잡도(Model Complexity)는 정규화와 밀접하게 관련되지만, 복잡도를 단순히 매개변수의 개수로 측정할 수 있는 것은 아니다. 동일한 매개변수 수를 가진 두 신경망도 구조(Architecture), 매개변수 분포(Parameter Distribution), 학습 절차(Training Procedure)에 따라 상당히 다른 일반화 성능을 보일 수 있다. 정규화는 학습 알고리즘이 생성할 가능성이 높은 함수나 해를 제한함으로써 유효 모델 용량(Effective Model Capacity)에 영향을 준다.

편향-분산 관점(Bias-Variance Perspective)은 정규화가 유용한 이유를 설명하는 또 다른 해석을 제공한다. 유연성이 높은 모델은 낮은 학습 편향(Training Bias)을 보일 수 있지만, 학습 데이터셋의 변화에 대한 민감도가 클 수 있다. 적절한 정규화를 도입하면 이러한 민감도를 줄여 기대 예측 성능(Expected Prediction Performance)을 개선할 가능성이 있다. 그러나 제약이 지나치면 편향(Bias)이 증가하여 정확한 예측에 필요한 관계를 표현하지 못할 수 있다.

잡음(Noise)은 실제 기계학습에서 피할 수 없는 고려 사항이다. 학습 데이터셋에는 측정 부정확성(Measurement Inaccuracy), 일관되지 않은 레이블(Inconsistent Label), 센서 교란(Sensor Disturbance), 표본 추출 인공물(Sampling Artifact) 등이 포함될 수 있다. 적절한 제어가 없다면 모델은 이러한 불규칙성을 의미 있는 신호(Signal)로 학습할 수 있다. 정규화는 이러한 현상을 억제하지만, 체계적인 레이블 오류(Systematic Labeling Error)나 근본적으로 부족한 데이터 품질(Data Quality)을 자동으로 해결하지는 못한다.

제한된 학습 데이터(Limited Training Data)는 여러 모델이 동일한 관측 결과를 설명할 수 있기 때문에 추가적인 문제를 발생시킨다. 이러한 불확실성(Uncertainty)은 가능한 관계의 수가 급격하게 증가하는 고차원 특징 공간(High-Dimensional Feature Space)에서 특히 중요해진다. 정규화는 특정 해를 선호하는 귀납적 선호(Inductive Preference)를 도입하여 모델 선택(Model Selection)의 모호성을 줄이고, 관측 데이터가 부족한 상황에서 성능을 개선할 가능성을 높인다.

정규화는 제약 최적화(Constrained Optimization)의 관점에서도 해석할 수 있다. 손실 함수에 벌점을 추가하는 대신 모델 매개변수가 지정된 제약 조건을 만족하도록 하면서 경험적 오차(Empirical Error)를 최소화할 수 있다. 적절한 수학적 조건에서는 벌점 기반 정식화(Penalized Formulation)와 제약 기반 정식화(Constrained Formulation)가 밀접하게 연결된다. 이러한 관점은 정규화가 허용되거나 선호되는 해 공간(Solution Space)을 제한하는 방식임을 보여준다.

베이지안 관점(Bayesian Perspective)에서 특정 정규화 벌점은 매개변수의 사전 분포(Prior Distribution)에 대한 가정과 대응된다. 예를 들어 이차 벌점(Quadratic Penalty)은 가우시안 사전 분포(Gaussian Prior)와 연결될 수 있으며, 절댓값 벌점(Absolute-Value Penalty)은 일반적인 정식화에서 라플라스 사전 분포(Laplace Prior)에 대응된다. 이러한 관계는 정규화가 단순한 수치적 조정이 아니라 타당한 해에 대한 사전 선호(Prior Preference)를 표현할 수 있음을 보여준다.

정규화가 명시적인 수학적 벌점에만 제한되는 것은 아니다. 학습 절차는 확률적 최적화(Stochastic Optimization), 매개변수 초기화(Parameter Initialization), 구조적 제약(Architectural Restriction), 학습 지속 시간(Training Duration) 등을 통해 일반화에 영향을 줄 수 있다. 이러한 메커니즘은 목적 함수에 추가적인 벌점이 명시되지 않더라도 학습 알고리즘이 특정 해를 선호하도록 만드는 암묵적 정규화(Implicit Regularization)를 도입할 수 있다.

확률적 경사 하강법(Stochastic Gradient Descent, SGD)은 최적화 동역학(Optimization Dynamics)의 중요성을 보여준다. 경사(Gradient)는 무작위로 선택된 미니배치(Mini-Batch)를 사용하여 추정되므로 매개변수 갱신(Parameter Update)에 확률적 변동(Stochastic Variation)이 포함된다. 이러한 변동은 손실 지형(Loss Landscape)에서 탐색되는 영역과 최종적으로 선택되는 해에 영향을 줄 수 있다. 그러나 확률성이 항상 일반화를 개선하는 것은 아니며, 그 효과는 학습률(Learning Rate), 배치 크기(Batch Size), 모델 특성에 따라 달라진다.

표현력이 높은 신경망(Expressive Neural Network)을 다양성이 제한된 데이터셋으로 학습할 때 정규화는 특히 중요하다. 모델은 높은 학습 정확도(Training Accuracy)를 달성하더라도 익숙하지 않은 조명, 시점(Viewpoint), 환경 조건에서 객체를 인식하지 못할 수 있다. 불변성(Invariance)을 장려하거나 의미 있는 입력 변형(Input Variation)에 모델을 노출하는 기법은 원래 관측 데이터의 범위를 넘어 패턴을 인식하는 능력을 개선할 수 있다.

컴퓨터 비전(Computer Vision)에서 정규화는 모델이 본질적인 시각적 특성(Essential Visual Characteristic)과 우연히 나타나는 이미지 속성(Incidental Image Property)을 구분하도록 돕는다. 분류 신경망(Classification Network)은 방향, 조명, 배경이 합리적인 범위에서 변화하더라도 객체를 인식하는 것이 바람직하다. 관련 없는 시각적 세부 사항에 대한 의존성을 줄이는 학습 전략은 변환(Transformation)이 예측 작업에 필요한 정보를 보존한다는 조건에서 강건성을 향상할 수 있다.

자연어 처리(Natural Language Processing, NLP)에서도 모델이 단어와 레이블 사이의 피상적인 상관관계(Superficial Correlation)를 활용할 수 있기 때문에 유사한 문제가 발생한다. 예를 들어 감성 분류기(Sentiment Classifier)는 문맥적 의미(Contextual Meaning)를 이해하지 못한 상태에서 특정 표현을 긍정적인 결과와 연결할 수 있다. 적절한 정규화는 불안정한 상관관계에 대한 의존성을 줄일 수 있지만, 강건한 언어 이해(Robust Language Understanding)를 위해서는 대표성 있는 데이터와 적절한 모델 구조도 필요하다.

로보틱스(Robotics)와 자율 시스템(Autonomous System)에서 정규화의 중요성은 일반적인 벤치마크 정확도(Benchmark Accuracy)를 넘어선다. 인식 및 예측 모델(Perception and Prediction Model)은 변화하는 환경 조건, 센서 불확실성(Sensor Uncertainty), 다양한 운영 시나리오(Operating Scenario)에서 동작해야 한다. 실험실 조건만 암기한 모델은 실제 배포(Deployment) 환경에서 예측하기 어려운 동작을 보일 수 있다. 정규화는 학습 환경에만 존재하는 세부 특성에 대한 불필요한 의존성을 억제하여 신뢰성(Reliability)에 기여한다.

정규화와 강건성(Robustness)의 관계는 신중하게 해석해야 한다. 정규화된 모델은 동일한 분포에서 추출한 새로운 샘플에 대해 더 나은 일반화 성능을 보일 수 있지만, 상당한 분포 이동(Distribution Shift)이 발생했을 때도 안정적으로 작동한다는 보장은 없다. 센서 특성, 운영 환경, 작업 요구사항의 변화는 일반적인 정규화만으로 해결하기 어려운 문제를 유발할 수 있으며, 추가적인 적응(Adaptation)이나 검증(Validation)이 필요할 수 있다.

현대 심층학습(Modern Deep Learning)은 모델 복잡도와 과적합에 대한 전통적인 가정을 더욱 복잡하게 만든다. 과매개변수화된 신경망(Overparameterized Neural Network)은 학습 오차가 거의 0에 도달하면서도 높은 테스트 성능(Test Performance)을 유지하는 경우가 있다. 이러한 현상은 일반화가 단순히 매개변수 수에만 의존하지 않고 데이터 구조(Data Structure), 모델 구조, 최적화 동역학, 학습 과정에서 선택된 해의 특성에 의해 결정된다는 사실을 보여준다.

따라서 정규화를 모든 모델의 크기를 줄이기 위한 요구사항으로만 이해해서는 안 된다. 정규화의 더 넓은 목적은 유용한 일반화 특성(Generalization Characteristic)을 가진 해를 향해 학습을 유도하는 것이다. 일부 정규화 전략(Regularization Strategy)은 매개변수 크기를 제한하지만, 다른 전략은 학습 샘플을 수정하거나 확률적 동작(Stochastic Behavior)을 도입하고, 학습 시간을 제한하거나 더 부드러운 예측 분포(Predictive Distribution)를 장려한다.

정규화의 효과는 모델 매개변수를 학습하는 데 사용되지 않은 데이터를 통해 평가해야 한다. 검증 데이터셋(Validation Dataset)은 정규화 강도(Regularization Strength)를 선택하고 대안적인 전략을 비교하기 위한 실용적인 기준을 제공한다. 그러나 동일한 검증 데이터셋에 대해 반복적으로 조정하면 선택 편향(Selection Bias)이 발생할 수 있다. 따라서 최종 모델 성능에 대한 편향이 적은 추정치를 얻기 위해 별도의 테스트 데이터셋(Test Dataset)이 중요하다.

학습 및 검증 학습 곡선(Training and Validation Learning Curve)은 정규화가 적절하게 설정되었는지 파악하는 데 도움이 된다. 학습 성능과 검증 성능 사이의 차이가 크면 과적합을 의미할 수 있으며, 두 데이터셋 모두에서 성능이 낮으면 모델 용량 부족(Insufficient Model Capacity), 최적화 문제(Optimization Difficulty), 과도한 정규화(Excessive Regularization)를 의심할 수 있다. 그러나 이러한 패턴은 확정적인 원인이 아니라 진단 지표(Diagnostic Indicator)이므로 추가적인 실험이 필요한 경우가 많다.

정규화는 학습 데이터의 양과 다양성(Data Diversity)에도 영향을 받는다. 데이터셋이 작을 때는 관측 결과가 실제 데이터 분포에 대한 제한적인 정보만 제공하므로 더 강한 귀납적 제약(Inductive Constraint)이 유용할 수 있다. 데이터셋이 커지고 대표성(Representativeness)이 높아지면 특정 제약의 필요성이 달라질 수 있다. 그러나 데이터셋이 커진다고 해서 과적합이 자동으로 사라지거나 강건성이 보장되는 것은 아니다.

정규화의 계산적 영향(Computational Consequence)은 기법에 따라 다르다. 일부 벌점은 추가적인 처리 비용(Processing Cost)이 상대적으로 작지만, 다른 기법은 학습 파이프라인(Training Pipeline)을 수정하거나 추가적인 데이터 변환(Data Transformation)을 요구한다. 정규화는 수렴 속도(Convergence Speed)와 만족스러운 성능에 도달하는 데 필요한 학습 반복 횟수(Training Iteration)에도 영향을 줄 수 있다. 따라서 실제 기법 선택에서는 예측 품질, 계산 자원(Computational Resource), 운영 요구사항(Operational Requirement)을 함께 고려해야 한다.

정규화와 최적화(Optimization) 사이에는 중요한 차이가 존재한다. 최적화는 목적 함수를 감소시키기 위해 모델 매개변수를 어떻게 갱신할지 결정하는 반면, 정규화는 어떤 해를 선호할지 또는 학습을 어떻게 제한할지에 영향을 준다. 두 과정은 밀접하게 상호작용하며, 정규화 전략의 변경은 손실 지형, 경사 동작(Gradient Behavior), 수렴 경로(Convergence Trajectory), 최종 매개변수 구성(Parameter Configuration)을 변화시킬 수 있다.

정규화의 핵심 목적은 모델이 학습 과정에서 습득한 내용과 이전에 관측하지 못한 데이터에 직면했을 때 나타나는 동작 사이의 관계를 개선하는 것이다. 성공적인 정규화는 의미 있는 예측 구조(Predictive Structure)를 보존하면서 학습 데이터셋의 우연한 특성에 대한 의존성을 제한한다. 이러한 원리를 이해하는 것은 명시적 벌점(Explicit Penalty), 확률적 방법(Stochastic Method), 데이터 증강(Data Augmentation), 조기 종료(Early Stopping), 현대적 정규화 전략(Modern Regularization Strategy)을 학습하기 위한 기초가 된다.

## 03.01 L1 L2 Regularization

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

L1 및 L2 정규화(L1 and L2 Regularization)는 기계학습(Machine Learning)에서 모델 복잡도(Model Complexity)를 제어하고 일반화 성능(Generalization Performance)을 향상하기 위한 핵심 기법이다. 두 방법 모두 모델 매개변수(Model Parameter)에 벌점(Penalty)을 추가하여 학습 목적 함수(Training Objective)를 수정하며, 불필요하게 큰 계수(Coefficient)에 의존하는 해(Solution)를 억제한다. 두 방법은 과적합(Overfitting)을 줄인다는 공통된 목적을 가지지만, 서로 다른 수학적 제약(Mathematical Constraint)을 적용하므로 매개변수 분포(Parameter Distribution)에 미치는 영향도 다르다.

L1 정규화(L1 Regularization)는 모델 매개변수의 절댓값 합(Sum of Absolute Values)에 비례하는 벌점을 추가한다. 목적 함수(Objective Function)는 J(θ) = L(θ) + λΣ\|θᵢ\|로 표현할 수 있으며, 여기서 L(θ)는 기존 손실 함수(Loss Function)를 나타내고 λ는 정규화 강도(Regularization Strength)를 제어한다. 벌점은 매개변수 크기에 비례하여 선형적으로 증가하므로 L1 정규화는 일부 계수가 정확히 0이 되는 희소 해(Sparse Solution)를 유도한다.

희소성(Sparsity)은 L1 정규화의 가장 중요한 특성이다. 특정 계수를 제거함으로써 이 방법은 암묵적인 특징 선택(Implicit Feature Selection)을 수행하고 더 적은 입력 변수(Input Variable)에 의존하는 모델을 생성할 수 있다. 이러한 특성은 데이터셋(Dataset)에 관련성이 낮거나 중복된 특징(Feature)이 많이 포함되어 있을 때 특히 유용하다. 그러나 예측 변수(Predictor) 사이의 상관관계(Correlation)가 높거나 학습 데이터셋이 변경되면 선택되는 특징이 달라질 수 있다.

L2 정규화(L2 Regularization)는 매개변수 값의 제곱합(Sum of Squared Parameter Values)에 비례하는 벌점을 도입한다. 정규화된 목적 함수(Regularized Objective)는 일반적으로 J(θ) = L(θ) + (λ/2)Σθᵢ²로 표현된다. L1 정규화와 달리 L2 정규화는 일반적으로 계수를 정확히 0으로 만들지 않고 연속적으로 0에 가깝게 축소한다. 따라서 명시적인 희소 매개변수 구성(Sparse Parameter Configuration)보다는 분산 표현(Distributed Representation)을 장려한다.

절댓값 벌점(Absolute-Value Penalty)과 제곱 벌점(Squared Penalty)의 수학적 차이는 두 방법의 상반된 동작을 설명한다. L1 정규화는 0이 아닌 계수에 일정한 크기의 벌점 기울기(Penalty Gradient)를 적용하지만, L2 정규화는 매개변수 값에 비례하는 기울기를 생성한다. 따라서 L2에서는 큰 계수가 더 강한 절대적 벌점 영향을 받으며, L1에서는 미분 불가능한 최적화 기하학(Nonsmooth Optimization Geometry)의 특성으로 작은 계수가 0에 도달할 수 있다.

기하학적 관점(Geometric Perspective)에서 L1 정규화는 2차원 공간에서 매개변수를 마름모 형태의 영역(Diamond-Shaped Region)으로 제한하는 반면, L2 정규화는 원형 영역(Circular Region)에 대응한다. L1 제약 영역의 모서리는 최적 손실 등고선(Optimal Loss Contour)과 좌표축(Coordinate Axis)에서 교차하는 경우가 많아 계수가 0이 되는 해를 생성한다. 반면 L2 제약의 부드러운 경계(Smooth Boundary)는 여러 계수가 함께 축소되는 해를 선호한다.

L1 정규화는 회귀 계수(Regression Coefficient)의 절댓값에 벌점을 부여하면서 예측 오차(Prediction Error)를 최소화하는 라쏘 회귀(Lasso Regression)와 밀접하게 관련된다. L2 정규화는 계수의 제곱값에 벌점을 부여하는 릿지 회귀(Ridge Regression)의 기초를 형성한다. 두 접근법은 로지스틱 회귀(Logistic Regression)와 신경망 학습(Neural Network Training)을 포함하여 미분 가능한 다른 학습 목적 함수에도 자연스럽게 확장될 수 있다.

정규화 계수(Regularization Coefficient) λ는 매개변수 크기가 최적화 과정(Optimization Process)에 얼마나 강하게 영향을 미치는지 결정한다. λ가 0에 가까워지면 목적 함수는 원래의 경험적 손실(Empirical Loss)과 점점 유사해진다. 반대로 λ가 증가하면 매개변수 제약이 강화되어 분산(Variance)이 감소할 수 있지만 편향(Bias)은 증가할 수 있다. 과도한 정규화는 모델이 의미 있는 관계를 학습하지 못하게 만들 수 있으므로 λ는 검증(Validation)을 통해 선택해야 한다.

특징 스케일링(Feature Scaling)은 선형 모델(Linear Model)에서 L1 또는 L2 정규화를 구현할 때 특히 중요하다. 입력 변수의 수치 범위(Numerical Range)가 크게 다르면 해당 계수들이 서로 다른 척도(Scale)에서 작동하게 된다. 모든 계수에 동일한 벌점을 적용하면 의도하지 않은 선호가 발생할 수 있다. 정규화 모델을 학습하기 전에 수치형 특징(Numerical Feature)을 표준화(Standardization)하면 벌점이 보다 일관되게 작용하도록 만들 수 있다.

실제 구현(Implementation)에서는 일반적으로 학습 가능한 가중치(Trainable Weight)에 정규화를 적용하고 편향 매개변수(Bias Parameter)는 제외한다. 편향 항(Bias Term)은 특징에 대한 민감도(Feature Sensitivity)보다는 기준 예측값(Baseline Prediction)을 제어하는 경우가 많으므로, 여기에 벌점을 적용하면 불필요한 제약이 발생할 수 있다. 그러나 이러한 관례가 모든 경우에 적용되는 것은 아니며, 프레임워크(Framework)마다 매개변수 그룹화(Parameter Grouping)와 정규화 메커니즘이 다르므로 구현 세부 사항을 확인해야 한다.

기본적인 넘파이(NumPy) 구현은 예측값(Prediction)을 계산하고 기존 손실을 평가한 다음 선택한 벌점을 추가하는 방식으로 시작한다. L1 정규화에서는 가중치 벡터(Weight Vector)의 절댓값을 이용하여 추가 항을 계산한다. L2 정규화에서는 가중치의 제곱값을 합산한 후 선택한 계수를 곱한다. 이렇게 구성된 목적 함수는 적절한 최적화 알고리즘(Optimization Algorithm)을 사용하여 최소화한다.

경사 기반 최적화(Gradient-Based Optimization)에서는 L1 정규화를 구현할 때 특별한 주의가 필요하다. 절댓값 함수(Absolute-Value Function)는 0에서 미분할 수 없으므로 일반적인 기울기 대신 부분 기울기(Subgradient) 또는 특수한 최적화 절차(Specialized Optimization Procedure)를 사용한다. 0이 아닌 계수에서 L1 부분 기울기는 해당 계수의 부호(Sign)와 같다. 계수가 0인 경우에는 −1부터 1까지의 구간에 속하는 모든 값이 유효한 부분 기울기가 된다.

L2 정규화는 제곱 벌점이 모든 지점에서 미분 가능(Differentiable)하므로 상대적으로 구현이 간단하다. 목적 함수가 J(θ) = L(θ) + (λ/2)Σθᵢ²일 때 기울기는 ∇J(θ) = ∇L(θ) + λθ가 된다. 추가된 항은 최적화 과정에서 매개변수를 0에 가까운 방향으로 이동시킨다. 이러한 단순성 때문에 L2 정규화는 경사 기반 학습 알고리즘(Gradient-Based Learning Algorithm)에 특히 편리하게 적용할 수 있다.

표준 경사 하강법(Standard Gradient Descent)을 L2 정규화 목적 함수에 적용하면 매개변수 갱신(Parameter Update)은 θₜ₊₁ = θₜ − η(∇L(θₜ) + λθₜ)가 되며, 여기서 η는 학습률(Learning Rate)을 의미한다. 이를 정리하면 θₜ₊₁ = (1 − ηλ)θₜ − η∇L(θₜ)가 된다. 이 식은 일반적인 경사 하강법에서 L2 정규화와 곱셈 방식의 가중치 감쇠(Multiplicative Weight Decay)가 연결되는 원리를 보여준다.

가중치 감쇠(Weight Decay)와 L2 정규화는 표준 경사 하강법에서 계수를 적절히 일치시키면 동등한 효과를 나타낸다. 그러나 벌점 기울기가 적응형 모멘트 추정(Adaptive Moment Estimate)에 포함되는 경우에는 아담(Adam)과 같은 적응형 최적화 알고리즘(Adaptive Optimizer)에서 이러한 동등성이 일반적으로 성립하지 않는다. 아담W(AdamW)와 같은 최적화 알고리즘이 구현하는 분리형 가중치 감쇠(Decoupled Weight Decay)는 손실 기울기(Loss Gradient)와 별도로 매개변수 축소(Parameter Shrinkage)를 적용한다.

파이토치(PyTorch)에서는 계산된 학습 손실(Training Loss)에 가중치 절댓값의 합을 추가하여 명시적 L1 정규화(Explicit L1 Regularization)를 구현할 수 있다. 자동 미분(Automatic Differentiation)은 미분 불가능한 지점에서 정의된 동작을 사용하여 해당 기울기 계산을 처리한다. L2 방식의 정규화는 명시적인 가중치 제곱 벌점(Squared-Weight Penalty)을 추가하여 구현할 수 있으며, 최적화 알고리즘의 가중치 감쇠 기능도 일반적으로 사용되는 구현 방법이다.

일반적인 파이토치 학습 루프(PyTorch Training Loop)는 먼저 순전파(Forward Pass)를 수행하고 작업 손실(Task Loss)을 계산한 다음 선택한 모델 매개변수로부터 정규화 벌점(Regularization Penalty)을 구성한다. 이후 결합된 손실(Combined Loss)에 대해 역전파(Backward Propagation)를 수행하고 최적화 알고리즘이 가중치를 갱신한다. 이러한 접근법을 사용하면 개발자는 정규화를 적용할 계층(Layer)을 제어하고 서로 다른 매개변수 그룹(Parameter Group)에 다른 벌점 강도를 적용할 수 있다.

사이킷런(Scikit-learn)은 라쏘(Lasso), 릿지(Ridge), 엘라스틱넷(ElasticNet)과 같은 추정기(Estimator)를 통해 정규화된 선형 모델의 검증된 구현을 제공한다. 이러한 추정기는 최적화 절차를 내부적으로 처리하므로 사용자는 특징 준비(Feature Preparation), 하이퍼파라미터 선택(Hyperparameter Selection), 평가(Evaluation)에 집중할 수 있다. 다만 사용자 정의 구현(Custom Implementation)과 결과를 비교하기 전에 정규화 매개변수와 목적 함수 스케일링(Objective Scaling)의 규칙을 확인해야 한다.

엘라스틱넷(Elastic Net)은 L1과 L2 벌점을 하나의 목적 함수에 결합한다. 정규화 항은 λ₁Σ\|θᵢ\| + (λ₂/2)Σθᵢ²로 표현할 수 있으며, 이를 통해 희소성과 연속적인 계수 축소(Continuous Coefficient Shrinkage)를 동시에 적용한다. 이러한 조합은 예측 변수 사이에 상관관계가 존재할 때 유용하며, 많은 실제 상황에서 순수한 L1 정규화보다 관련된 특징 그룹(Feature Group)을 효과적으로 유지할 수 있다.

L1과 L2 중 어떤 방법을 선택할지는 학습 목적(Learning Objective)과 데이터 특성(Data Characteristic)에 따라 달라진다. 특징 선택, 간결한 표현(Compact Representation), 계수 해석 가능성(Coefficient Interpretability)이 중요하다면 L1이 유용하다. 반면 많은 특징이 유용한 정보를 제공하고 안정적인 계수 축소가 필요하다면 L2가 선호되는 경우가 많다. 어떤 방법도 모든 데이터셋에서 항상 우수한 성능을 보장하지는 않는다.

고차원 분류 문제(High-Dimensional Classification Problem)에서 L1 정규화는 관련성이 낮은 입력 차원(Input Dimension)에 대한 의존성을 줄일 수 있다. 예를 들어 텍스트 분류 모델(Text Classification Model)에는 수천 개의 희소 어휘 특징(Sparse Vocabulary Feature)이 포함될 수 있지만, 그중 일부만 예측에 의미 있게 기여한다. L1은 불필요한 계수를 억제하여 해석 가능성을 개선하고 실제로 활성화된 특징(Active Feature)의 수를 줄일 가능성이 있다.

L2 정규화는 연결을 명시적으로 제거하지 않으면서 과도한 매개변수 크기를 억제하므로 신경망에서 널리 사용된다. 작은 가중치는 특정 입력 변동(Input Variation)에 대한 민감도를 줄이고 학습된 함수의 복잡도에 영향을 줄 수 있다. 그러나 매개변수 크기만으로 신경망의 일반화를 완전히 설명할 수는 없으며, 특히 정규화 계층(Normalization Layer)이나 스케일 불변 매개변수화(Scale-Invariant Parameterization)를 포함한 구조에서는 이러한 한계가 더욱 중요하다.

구현 품질(Implementation Quality)은 정규화 강도와 손실 감소 규칙(Loss Reduction Convention) 사이의 관계에 영향을 받는다. 원래 손실을 샘플 전체에 대해 평균화하지만 벌점은 매개변수 전체에 대해 합산한다면, 매개변수 수의 변화가 두 항의 상대적 기여도(Relative Contribution)에 상당한 영향을 줄 수 있다. 마찬가지로 샘플 손실을 평균하지 않고 합산하면 계수를 적절히 조정하지 않는 한 λ의 실질적인 의미가 달라진다.

정규화 실험(Regularization Experiment)은 일관된 데이터 분할(Data Partition)과 평가 절차(Evaluation Procedure)를 사용해야 한다. 실용적인 비교 방법은 정규화를 적용하지 않은 기준 모델(Unregularized Baseline), L1 정규화 모델, L2 정규화 모델을 유사한 조건에서 학습하는 것이다. 학습 손실, 검증 성능(Validation Performance), 계수 크기(Coefficient Magnitude), 0인 매개변수의 개수는 각 방법이 학습에 미치는 영향을 상호 보완적으로 설명한다.

하이퍼파라미터 튜닝(Hyperparameter Tuning)은 일반적으로 로그 간격(Logarithmic Range)에 따라 여러 정규화 강도를 평가한다. 작은 계수는 모델에 더 큰 유연성(Flexibility)을 허용하지만, 큰 계수는 더 강한 제약을 부과한다. 데이터셋이 제한적일 때 교차 검증(Cross-Validation)은 더 안정적인 성능 추정치를 제공할 수 있다. 다만 정보 누출(Information Leakage)을 방지하려면 각 학습 폴드(Training Fold) 내부에서 전처리(Preprocessing)를 독립적으로 학습해야 한다.

중요한 구현 고려 사항은 정규화를 강화한다고 해서 반드시 일반화 성능이 향상되는 것은 아니라는 점이다. 초기에는 분산 감소(Reduced Variance)를 통해 성능이 개선될 수 있지만, 벌점 강도를 계속 높이면 유용한 예측 정보(Predictive Information)까지 제거될 수 있다. 학습 동작(Training Behavior)과 함께 검증 지표(Validation Metric)를 모니터링하면 추가적인 정규화가 오히려 성능을 저하시키기 시작하는 지점을 파악하는 데 도움이 된다.

L1 및 L2 정규화는 기계학습에서 명시적인 복잡도 제어(Explicit Complexity Control)를 이해하기 위한 실용적인 기초를 제공한다. 두 방법의 수학적 차이는 L1이 희소 해를 선호하고 L2가 연속적인 매개변수 축소를 장려하는 이유를 설명한다. 올바른 구현을 위해서는 목적 함수 스케일링, 특징 정규화(Feature Normalization), 매개변수 선택(Parameter Selection), 최적화 알고리즘의 동작(Optimizer Behavior), 검증 절차에 주의를 기울여야 하며, 이를 통해 정규화가 단순히 계수 크기를 줄이는 데 그치지 않고 일반화 성능을 실질적으로 개선하도록 해야 한다.

## 03.02 Dropout

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

드롭아웃(Dropout)은 학습 과정에서 선택된 신경망 뉴런(Neuron)을 무작위로 비활성화하여 과적합(Overfitting)을 줄이는 데 사용되는 대표적인 심층학습(Deep Learning) 정규화 기법(Regularization Technique)이다. 모든 뉴런이 매번 순전파(Forward Pass)에 참여하도록 하는 대신, 드롭아웃은 신경망의 내부 표현(Internal Representation)에 확률적 변동(Stochastic Variation)을 도입한다. 이러한 메커니즘은 개별 뉴런에 대한 과도한 의존성을 억제하고 더욱 분산되고 강건한 특징 표현(Feature Representation)을 학습하도록 유도한다.

드롭아웃의 근본적인 목적은 뉴런 사이의 공동 적응(Co-adaptation)을 줄이는 것이다. 공동 적응은 여러 뉴런이 정확한 예측을 생성하기 위해 서로 지나치게 의존할 때 발생한다. 이러한 의존 관계는 학습 데이터(Training Data)에 효과적으로 적합할 수 있지만, 관측되지 않은 데이터(Unseen Data)에 대한 일반화(Generalization)는 저하될 수 있다. 드롭아웃은 학습 중 뉴런의 기여를 무작위로 제거함으로써 남아 있는 뉴런들이 다양한 신경망 구성(Network Configuration)에서도 유용한 특징을 학습하도록 유도한다.

드롭아웃은 각 학습 순전파 과정에서 선택된 활성화 값(Activation)에 대한 이진 마스크(Binary Mask)를 샘플링(Sampling)하는 방식으로 작동한다. 각 요소는 유지 확률(Retention Probability) 1 − p에 따라 독립적으로 유지되거나 드롭아웃 확률(Dropout Probability) p에 따라 0으로 설정된다. 결과적으로 활성화 패턴(Activation Pattern)은 학습 반복(Training Iteration)마다 달라지며, 동일한 학습 가능 매개변수(Trainable Parameter)를 유지하면서 여러 무작위 부분 신경망(Subnetwork)을 경험하게 된다.

역 드롭아웃(Inverted Dropout)의 수학적 표현(Mathematical Formulation)은 y = (m ⊙ x)/(1 − p)로 나타낼 수 있다. 여기서 x는 원래의 활성화 텐서(Activation Tensor), m은 유지 확률이 1 − p인 베르누이 마스크(Bernoulli Mask), ⊙는 요소별 곱셈(Elementwise Multiplication)을 의미한다. 유지된 활성화 값을 1 − p로 나누면 p가 1보다 작은 조건에서 학습 중 활성화 값의 기댓값(Expected Value)을 보존할 수 있다.

역 드롭아웃은 현대 심층학습 프레임워크(Deep Learning Framework)에서 표준적으로 사용되는 구현 전략(Implementation Strategy)이다. 학습 중에는 무작위로 제거된 뉴런을 보상하기 위해 유지된 활성화 값의 크기를 조정한다. 추론(Inference) 단계에서는 드롭아웃을 비활성화하고 추가적인 스케일링(Scaling) 없이 활성화 값을 전달한다. 이러한 방식은 드롭아웃 마스크를 샘플링하거나 활성화 크기를 수정하지 않고 동일한 결정론적 신경망(Deterministic Network)으로 예측할 수 있으므로 배포(Deployment)를 단순화한다.

드롭아웃 확률은 확률적 정규화(Stochastic Regularization)의 강도를 제어한다. 작은 확률은 상대적으로 적은 활성화 값을 제거하여 제한적인 교란(Perturbation)을 도입하지만, 큰 확률은 정보 흐름(Information Flow)에 더 강한 변화를 발생시킨다. 지나치게 높은 드롭아웃 확률은 신경망이 중요한 관계를 학습하지 못하게 하여 과소적합(Underfitting)이나 불안정한 수렴(Unstable Convergence)을 유발할 수 있다. 적절한 확률은 모델 구조(Architecture), 데이터셋 크기(Dataset Size), 학습 조건(Training Condition)에 따라 달라진다.

드롭아웃은 종종 모델 평균화(Model Averaging)의 근사적인 형태로 해석된다. 샘플링된 각 마스크는 서로 다른 유효 부분 신경망(Effective Subnetwork)을 생성하며, 학습 과정에서는 이러한 구성 사이에 매개변수를 공유한다. 추론 단계에서는 전체 신경망이 여러 부분 신경망에서 학습된 정보를 결합하는 계산 효율적인 근사(Computationally Efficient Approximation)를 제공한다. 그러나 이러한 해석은 근사적인 것이며 가능한 모든 모델의 예측을 평균하는 것과 수학적으로 완전히 동등하지는 않다.

완전 연결 신경망(Fully Connected Neural Network)에서는 일반적으로 선택된 은닉 계층(Hidden Layer)의 활성화 함수(Activation Function) 뒤에 드롭아웃을 적용한다. 이 연산은 중간 특징 벡터(Intermediate Feature Vector)의 일부 요소를 무작위로 억제한 후 다음 계층으로 전달한다. 이러한 배치는 개별 은닉 표현(Hidden Representation)에 대한 의존성을 줄일 수 있지만, 최적의 적용 위치는 모델 구조의 세부 사항과 정규화 계층(Normalization Layer) 또는 잔차 연결(Residual Connection)과의 상호작용에 따라 달라진다.

드롭아웃은 L1 및 L2와 같은 가중치 기반 정규화(Weight-Based Regularization) 방법과 다르게 작동한다. 가중치 벌점(Weight Penalty)은 목적 함수(Objective Function)에 추가적인 항을 도입하여 매개변수 크기에 직접 영향을 주지만, 드롭아웃은 학습 중 계산 경로(Computational Pathway)를 변경한다. 드롭아웃의 정규화 효과는 명시적인 계수 축소(Coefficient Shrinkage)가 아니라 확률적인 활성화 억제(Stochastic Activation Suppression)에서 발생한다. 이러한 방법들은 결합할 수 있지만 상호작용 효과는 실험적으로 검증해야 한다.

기본적인 드롭아웃 구현은 넘파이(NumPy)를 사용하여 활성화 배열(Activation Array)과 동일한 형태의 무작위 이진 마스크를 생성하는 방식으로 구성할 수 있다. 각 마스크 요소는 유지 확률에 따라 샘플링되며, 활성화 배열에 마스크를 요소별로 곱한다. 이후 결과를 유지 확률로 나누면 역 드롭아웃이 구현되고 활성화 값의 기대 척도(Expected Activation Scale)를 유지할 수 있다.

드롭아웃을 수동으로 구현할 때는 학습 모드(Training Mode)와 추론 모드(Inference Mode)를 구분해야 한다. 학습 중에는 순전파 과정에서 새로운 마스크를 샘플링하고 확률적인 억제를 적용한다. 반면 추론 중에는 원래 활성화 값을 변경하지 않고 반환한다. 구현 과정에서는 경계 조건(Boundary Condition)도 처리해야 하며, 드롭아웃 확률이 0을 포함하고 1은 제외하는 유효 구간(Valid Interval)에 있도록 보장해야 한다.

파이토치(PyTorch)는 학습 중 역 드롭아웃을 자동으로 적용하는 torch.nn.Dropout을 제공한다. 예를 들어 nn.Dropout(p=0.5)를 사용하여 드롭아웃 계층(Dropout Layer)을 생성할 수 있으며, 이 확률은 제거될 것으로 예상되는 활성화 요소의 비율을 지정한다. model.train()을 통해 모델을 학습 모드로 설정하면 드롭아웃 계층은 순전파 실행 중 무작위 마스크(Random Mask)를 생성한다.

파이토치는 model.eval()을 통해 모델이 평가 모드(Evaluation Mode)에 진입하면 표준 드롭아웃(Standard Dropout)을 자동으로 비활성화한다. 이러한 구분은 안정적인 검증(Validation) 및 추론 결과를 얻는 데 필수적이다. 드롭아웃이 활성화된 상태에서 평가를 수행하면 동일한 입력에 대해 반복적인 순전파를 실행할 때 예측 결과가 달라질 수 있다. 따라서 올바른 모드 관리(Mode Management)는 신뢰할 수 있는 드롭아웃 구현의 기본 요구사항이다.

일반적인 파이토치 다층 퍼셉트론(Multilayer Perceptron, MLP)은 선형 변환(Linear Transformation), 비선형 활성화 함수(Nonlinear Activation Function), 드롭아웃 계층을 결합한다. 예를 들어 은닉 표현은 nn.Linear, nn.ReLU, nn.Dropout을 차례로 통과한 후 다음 변환 단계로 전달될 수 있다. 학습 중 드롭아웃 계층은 활성화 요소를 무작위로 억제하며, 자동 미분(Automatic Differentiation)은 유지된 계산 경로를 통해 기울기(Gradient)를 전파한다.

역전파(Backward Propagation) 과정에서 드롭아웃으로 제거된 활성화 값은 해당 반복에서 마스킹된 경로를 통한 기울기 계산에 기여하지 않는다. 유지된 활성화 값은 순전파에서 사용한 스케일링 계수(Scaling Factor)에 따라 기울기를 전파한다. 반복마다 서로 다른 마스크가 샘플링되므로 개별 뉴런은 활성화된 신경망 구성 요소(Active Network Component)의 조합이 달라지는 조건에서 학습 신호(Learning Signal)를 받는다.

텐서플로(TensorFlow)와 케라스(Keras)는 tf.keras.layers.Dropout 계층을 통해 동일한 기능을 제공한다. 드롭아웃 비율(Dropout Rate)은 학습 중 입력 요소를 0으로 설정할 확률을 결정한다. 케라스는 표준 학습 및 예측 인터페이스(Training and Prediction Interface)를 사용할 때 학습과 추론 사이의 차이를 자동으로 관리하므로 수동 마스크 처리(Manual Mask Handling)의 필요성을 줄여준다.

합성곱 신경망(Convolutional Neural Network, CNN)에서는 드롭아웃 적용 위치(Dropout Placement)에 특별한 주의가 필요하다. 표준 요소별 드롭아웃(Elementwise Dropout)은 활성화 요소를 독립적으로 억제하지만, 인접한 공간 활성화(Spatial Activation)에는 강한 상관관계가 존재할 수 있다. 공간 드롭아웃(Spatial Dropout)의 변형 기법은 전체 특징 채널(Feature Channel) 또는 구조화된 활성화 그룹(Structured Activation Group)을 억제하여 합성곱 특징 맵(Convolutional Feature Map)에서 더욱 의미 있는 정규화 효과를 제공할 수 있다.

파이토치의 nn.Dropout2d와 nn.Dropout3d는 적절한 합성곱 텐서(Convolutional Tensor)에 대해 채널 중심 드롭아웃(Channel-Oriented Dropout)을 지원한다. 이러한 계층은 모든 공간 활성화 값을 독립적으로 제거하는 대신 구조화된 특징 차원(Structured Feature Dimension)에 마스크를 적용한다는 점에서 일반 드롭아웃과 다르다. 적절한 변형 기법을 선택하려면 텐서 배치 구조(Tensor Layout), 공간적 의존성(Spatial Dependency), 의도한 정규화 동작(Regularization Behavior)을 이해해야 한다.

드롭아웃은 순환 및 시퀀스 처리 구조(Recurrent and Sequence-Processing Architecture)에서도 사용되지만, 구현에는 추가적인 주의가 필요하다. 모든 시간 단계(Time Step)에서 독립적으로 마스크를 샘플링하면 특정 순환 계산(Recurrent Computation)에 바람직하지 않은 시간적 잡음(Temporal Noise)이 발생할 수 있다. 변분 드롭아웃(Variational Dropout)과 관련 방법은 모델 구조와 구현 방식에 따라 시퀀스 위치(Sequence Position) 또는 시간 단계 사이에서 동일한 마스크를 재사용할 수 있다.

트랜스포머(Transformer) 구조는 어텐션 확률(Attention Probability), 잔차 경로(Residual Pathway), 피드포워드 표현(Feedforward Representation) 등 선택된 구성 요소에 드롭아웃을 적용한다. 그러나 드롭아웃 비율과 적용 위치는 모델 계열(Model Family) 및 학습 체계(Training Regime)에 따라 상당히 달라진다. 일부 대규모 사전 학습 트랜스포머(Large-Scale Pretrained Transformer)는 사전 학습(Pretraining) 과정에서 드롭아웃을 거의 사용하지 않거나 전혀 사용하지 않으며, 이는 성공적인 일반화를 위해 드롭아웃이 항상 필요한 것은 아님을 보여준다.

드롭아웃은 정규화 계층 및 다른 학습 메커니즘(Training Mechanism)과 상호작용한다. 배치 정규화(Batch Normalization)는 학습 배치(Training Batch)에서 활성화 통계(Activation Statistics)를 추정하지만, 드롭아웃은 활성화 분포(Activation Distribution)를 확률적으로 변화시킨다. 따라서 특정 계층 배치(Layer Arrangement)는 학습과 추론 사이의 동작 차이를 발생시킬 수 있다. 드롭아웃을 배치 정규화 또는 다른 정규화 방법과 결합할 때는 신중한 구조 설계(Architectural Design)와 검증이 필요하다.

드롭아웃과 데이터셋 크기의 관계는 실제 모델 개발(Model Development)에서 중요하다. 작은 데이터셋은 신경망이 제한된 관측 결과를 암기할 가능성이 높기 때문에 추가적인 확률적 정규화의 이점을 얻을 수 있다. 그러나 학습 샘플(Training Sample)이 이미 부족하거나 모델이 유용한 특징을 학습하는 데 어려움을 겪는 경우 과도한 드롭아웃은 오히려 해로울 수 있다. 따라서 더 강한 드롭아웃이 유익한지는 검증 성능(Validation Performance)을 기준으로 판단해야 한다.

드롭아웃은 각 학습 반복에서 서로 다른 유효 신경망 구성(Effective Network Configuration)을 사용하므로 최적화(Optimization)를 더 어렵게 만들 수 있다. 드롭아웃을 사용하지 않는 동일한 모델보다 학습 손실(Training Loss)이 느리게 감소할 수 있다. 그러나 검증 성능이 개선된다면 이러한 추가적인 최적화 어려움이 정당화될 수 있다. 모델 비교에서는 학습 손실만 고려하지 않고 수렴 동작(Convergence Behavior)과 일반화 성능을 함께 평가해야 한다.

실용적인 실험(Practical Experiment)에서는 동일한 신경망 구조(Neural Network Architecture)에 여러 드롭아웃 확률을 적용하여 학습 결과를 비교한다. 학습 절차에서는 데이터 분할(Data Partition), 최적화 알고리즘 설정(Optimizer Setting), 평가 지표(Evaluation Metric)를 가능한 한 동일하게 유지해야 한다. 학습 손실과 검증 손실(Validation Loss)을 모니터링하면 드롭아웃이 과적합을 줄이는지, 과소적합을 유발하는지, 또는 선택한 실험 조건에서 효과가 거의 없는지 확인할 수 있다.

드롭아웃은 몬테카를로 드롭아웃(Monte Carlo Dropout, MC Dropout)을 통해 불확실성 추정(Uncertainty Estimation)에도 활용될 수 있다. 이 접근법에서는 반복적인 추론 과정에서도 드롭아웃을 활성화하여 동일한 입력에 대해 여러 확률적 예측(Stochastic Prediction)을 생성한다. 예측 결과 사이의 변동은 예측 불확실성(Predictive Uncertainty)에 대한 근사적인 측정값을 제공하지만, 그 신뢰성은 모델 보정(Model Calibration), 드롭아웃 구성(Dropout Configuration), 근사에 사용된 가정(Approximation Assumption)에 따라 달라진다.

몬테카를로 드롭아웃은 일반적인 평가 모드에서 드롭아웃이 비활성화되므로 추론 동작(Inference Behavior)을 의도적으로 제어해야 한다. 구현에서는 다른 구성 요소, 특히 배치 정규화 계층을 평가 모드로 유지하면서 드롭아웃 계층만 선택적으로 활성화할 수 있다. 이후 반복적인 예측 결과를 예측 평균(Predictive Mean)과 분산(Variance) 등의 통계량으로 요약하여 단일 결정론적 출력(Deterministic Output)보다 많은 정보를 제공할 수 있다.

드롭아웃 구현은 배포 요구사항(Deployment Requirement)을 고려하여 평가해야 한다. 표준 추론에서는 일반적으로 드롭아웃을 비활성화하므로 추가적인 확률적 계산(Stochastic Computation)이 발생하지 않는다. 반면 몬테카를로 드롭아웃은 여러 차례의 순전파를 필요로 하므로 계산 지연(Computational Latency)이 증가한다. 실시간 시스템(Real-Time System)은 불확실성 추정의 이점과 처리량(Throughput), 메모리 사용량(Memory Consumption), 응답 시간 제약(Response-Time Constraint) 사이의 균형을 고려해야 한다.

드롭아웃은 명시적인 매개변수 벌점(Explicit Parameter Penalty)이 아니라 확률적 계산(Stochastic Computation)을 통해 정규화를 수행하는 중요한 사례이다. 그 효과는 적절한 확률 선택, 올바른 신경망 구성 요소에 대한 마스크 적용, 학습과 추론 동작의 정확한 분리에 따라 달라진다. 드롭아웃의 수학적 정식화(Mathematical Formulation)와 프레임워크 구현(Framework Implementation)을 함께 이해하면 과적합을 억제하면서 유용한 표현을 학습하는 신경망을 설계하기 위한 기초를 마련할 수 있다.

## 03.03 Data Augmentation

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

데이터 증강(Data Augmentation)은 학습 샘플(Training Sample)의 다양성을 확장하여 기계학습(Machine Learning) 모델의 일반화(Generalization) 성능을 향상하는 핵심 정규화 기법(Regularization Technique)이다. 완전히 새로운 관측 데이터(Observation)를 수집하는 대신, 기존 샘플에 제어된 변환(Controlled Transformation)을 적용하여 수정된 데이터를 생성한다. 이러한 변형은 모델이 개별 학습 사례(Training Instance)의 특정 특성을 암기하기보다 안정적인 패턴(Stable Pattern)을 학습하도록 유도한다.

데이터 증강의 핵심 원리는 선택된 변환 이후에도 의미 있는 정보(Meaningful Information)가 일관되게 유지되어야 한다는 것이다. 이미지 분류(Image Classification)에서는 밝기(Brightness), 방향(Orientation), 위치(Position)가 적절한 범위에서 변하더라도 객체의 정체성(Object Identity)이 유지될 수 있다. 모델을 이러한 변형에 노출하면 불필요한 변화에 대한 불변성(Invariance)을 학습하면서 올바른 예측을 결정하는 특징(Feature)에 대한 민감도(Sensitivity)는 유지하도록 유도할 수 있다.

데이터 증강은 독립적으로 수집된 관측 데이터의 수를 반드시 늘리지 않으면서 데이터셋(Dataset)의 유효 다양성(Effective Diversity)을 증가시킨다. 하나의 이미지에서 여러 변환 샘플(Transformed Sample)을 생성할 수 있지만, 이러한 샘플은 원본 데이터와 통계적으로 관련되어 있다. 따라서 원본 데이터셋에 중요한 클래스(Class), 환경 조건(Environmental Condition), 대표적인 운영 시나리오(Operating Scenario)가 부족하다면 데이터 증강만으로 추가적인 실제 데이터(Real-World Data)를 완전히 대체할 수 없다.

데이터 증강 전략(Augmentation Strategy)은 학습 작업(Learning Task)의 의미적 특성(Semantics)을 존중해야 한다. 수평 뒤집기(Horizontal Flipping)는 일반적인 객체 인식(Object Recognition)에는 적절할 수 있지만, 좌우 방향이 레이블(Label)을 결정하는 경우에는 부적절할 수 있다. 마찬가지로 임의 회전(Random Rotation)은 손글씨 숫자(Handwritten Digit)나 방향성 교통 표지판(Directional Traffic Sign)의 의미를 훼손할 수 있다. 유효한 데이터 증강은 목표 레이블(Target Label)을 보존하거나 수정된 입력에 맞추어 주석(Annotation)을 일관되게 변환해야 한다.

기하학적 데이터 증강(Geometric Augmentation)은 위치, 크기, 회전, 원근(Perspective)과 같은 공간적 속성(Spatial Property)을 수정한다. 대표적인 연산에는 무작위 자르기(Random Cropping), 수평 뒤집기, 이동(Translation), 크기 조정(Resizing), 아핀 변환(Affine Transformation)이 포함된다. 이러한 기법은 비전 모델(Vision Model)이 카메라 위치(Camera Placement)와 객체 외형(Object Appearance)의 변화에 대응하도록 돕는다. 그러나 과도한 변환은 중요한 특징을 제거하거나 비현실적인 샘플을 생성하여 학습 효과를 저하시킬 수 있다.

광도 기반 데이터 증강(Photometric Augmentation)은 객체의 기하학적 구조(Geometry)를 직접 수정하지 않고 이미지의 외형을 변경한다. 밝기, 대비(Contrast), 채도(Saturation), 색조(Hue), 색상 균형(Color Balance)을 조절하여 다양한 조명 및 카메라 조건(Camera Condition)을 모사할 수 있다. 이러한 변환은 조명 변화가 큰 실외 인식(Outdoor Perception)과 산업 검사(Industrial Inspection)에서 유용하다. 변환 강도는 물리적으로 타당한 운영 환경(Operating Environment)에 부합해야 한다.

잡음 주입(Noise Injection)은 학습 샘플에 제어된 교란(Perturbation)을 추가하여 측정 불확실성(Measurement Uncertainty)에 대한 내성을 향상한다. 가우시안 잡음(Gaussian Noise), 센서 잡음(Sensor Noise), 압축 인공물(Compression Artifact)은 실제 배포(Deployment) 과정에서 발생하는 불완전성을 근사할 수 있다. 그러나 무작위로 생성한 잡음이 실제 센서 동작(Sensor Behavior)을 항상 대표하는 것은 아니다. 효과적인 증강을 위해서는 사용하려는 데이터 획득 하드웨어(Acquisition Hardware)의 통계적 특성(Statistical Characteristic)을 이해하는 것이 중요하다.

가림 기반 데이터 증강(Occlusion-Based Augmentation)은 입력 이미지의 일부 영역을 무작위로 가려 모델이 불완전한 시각적 정보(Visual Evidence)에서도 객체를 인식하도록 유도한다. 컷아웃(Cutout)과 랜덤 이레이징(Random Erasing)은 학습 중 선택된 이미지 영역을 제거하거나 다른 값으로 대체한다. 이러한 방법은 특정 시각적 단서(Visual Cue)에 대한 의존성을 줄일 수 있지만, 지나친 가림은 클래스를 올바르게 구분하는 데 필요한 정보를 제거할 수 있다.

믹스업(Mixup)은 두 입력 데이터와 해당 목표값(Target)을 결합하여 합성 학습 샘플(Synthetic Training Sample)을 생성한다. 일반적인 수식은 x̃ = λx₁ + (1 − λ)x₂ 및 ỹ = λy₁ + (1 − λ)y₂로 표현되며, 여기서 λ는 혼합 분포(Mixing Distribution)에서 샘플링된다. 이 기법은 학습 샘플 사이에서 더 부드러운 전이(Smooth Transition)를 학습하도록 유도하고 과도하게 확신하는 예측(Overconfident Prediction)을 줄일 수 있지만, 적합성은 작업과 목표 표현(Target Representation)에 따라 달라진다.

컷믹스(CutMix)는 서로 다른 학습 이미지의 영역을 결합하면서 교체된 면적에 따라 목표값의 기여도(Target Contribution)를 조정한다. 픽셀값(Pixel Value)을 혼합하는 믹스업과 달리 컷믹스는 붙여 넣은 영역(Pasted Region)의 국소적인 시각 구조(Local Visual Structure)를 유지한다. 이 방법은 강건성(Robustness)을 개선하고 여러 객체 영역(Object Region)에 주의를 기울이도록 유도할 수 있다. 그러나 실제 객체 점유 영역(Object Occupancy)이 붙여 넣은 영역의 면적과 크게 다르면 목표값 혼합 규칙(Target Mixing Rule)이 부정확해질 수 있다.

객체 감지(Object Detection)를 위한 데이터 증강에서는 이미지와 경계 상자 주석(Bounding Box Annotation)을 동기화하여 변환해야 한다. 이미지의 크기 조정, 자르기, 뒤집기, 회전을 수행하면 동일한 기하학적 변환(Geometric Transformation)을 사용하여 경계 상자(Bounding Box)도 갱신해야 한다. 유효하지 않거나 잘린 경계 상자(Truncated Bounding Box)는 신중하게 처리해야 한다. 주석의 일관성(Annotation Consistency)을 유지하지 못하면 레이블 잡음(Label Noise)이 발생하고 위치 추정 정확도(Localization Accuracy)가 크게 저하될 수 있다.

의미론적 분할(Semantic Segmentation)에서도 픽셀 수준 마스크(Pixel-Level Mask)가 변환된 이미지와 정렬(Alignment)을 유지해야 하므로 유사한 요구사항이 적용된다. 기하학적 연산은 입력 이미지와 분할 마스크(Segmentation Mask)를 일관되게 수정해야 하지만, 광도 변환은 일반적으로 이미지에만 적용해야 한다. 보간 방법(Interpolation Method)도 중요한데, 범주형 분할 마스크(Categorical Segmentation Mask)는 중간 레이블값(Intermediate Label Value)을 생성하지 않고 이산적인 클래스 식별자(Discrete Class Identifier)를 유지해야 하기 때문이다.

실제 파이썬(Python) 구현에서는 torchvision.transforms 또는 다른 이미지 처리 라이브러리(Image-Processing Library)를 사용하여 데이터 증강 파이프라인(Augmentation Pipeline)을 구성할 수 있다. 일반적인 학습 파이프라인(Training Pipeline)은 무작위 크기 조정 자르기(Random Resized Cropping), 수평 뒤집기, 색상 조정(Color Adjustment), 텐서 변환(Tensor Conversion), 정규화(Normalization)를 결합한다. 이러한 연산은 순차적으로 실행되며 신경망(Neural Network)에 입력되기 전에 변환된 데이터를 생성한다.

파이토치(PyTorch)는 일반적으로 데이터셋 객체(Dataset Object) 내부 또는 데이터셋 클래스(Dataset Class)에 전달되는 변환 인자(Transform Argument)를 통해 데이터 증강을 적용한다. 샘플을 불러올 때 무작위 변환(Random Transformation)을 실행하면 모든 증강 이미지를 디스크(Disk)에 저장하지 않고도 새로운 학습 변형을 생성할 수 있다. 이러한 온라인 데이터 증강(Online Augmentation)은 저장 공간(Storage Requirement)을 줄이고 학습 에포크(Training Epoch)마다 모델이 서로 다른 변환을 경험하도록 한다.

실용적인 torchvision 파이프라인에서는 transforms.Compose를 사용하여 RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize와 같은 연산을 결합할 수 있다. 변환 순서(Transformation Order)는 일부 연산이 PIL 이미지(PIL Image)를 요구하는 반면 다른 연산은 텐서(Tensor)를 대상으로 작동하므로 중요하다. 개발자는 설치된 torchvision 버전(Version)과 데이터셋에서 사용하는 입력 표현(Input Representation)의 호환성(Compatibility)을 확인해야 한다.

학습 및 검증 파이프라인(Training and Validation Pipeline)은 신뢰할 수 있는 성능 측정(Performance Measurement)을 위해 분리해야 한다. 무작위 데이터 증강은 일반적으로 학습 샘플에만 적용하며, 검증 및 테스트 데이터(Validation and Test Data)에는 결정론적 전처리(Deterministic Preprocessing)를 사용한다. 이미지 분류에서는 크기 조정, 중앙 자르기(Center Cropping), 텐서 변환, 정규화를 검증 전처리에 사용하는 경우가 많다. 이러한 분리는 제어되지 않은 무작위 변환이 평가 결과에 영향을 주지 않도록 보장한다.

데이터 증강 강도(Augmentation Intensity)는 변환된 샘플이 원본에서 얼마나 멀어지는지 결정하는 중요한 하이퍼파라미터(Hyperparameter)이다. 약한 증강(Weak Augmentation)은 충분한 다양성을 제공하지 못할 수 있지만, 지나치게 강한 증강(Aggressive Augmentation)은 클래스 정체성(Class Identity)을 왜곡하거나 비현실적인 학습 조건을 도입할 수 있다. 따라서 강한 변환이 항상 더 나은 일반화를 제공한다고 가정하기보다 검증 실험(Validation Experiment)을 통해 적절한 강도를 선택해야 한다.

자동 데이터 증강(Automated Augmentation)은 사람이 직접 선택한 연산에만 의존하지 않고 유용한 변환 정책(Transformation Policy)을 탐색하는 방법이다. 오토어그먼트(AutoAugment)는 변환 조합(Transformation Combination)을 탐색하며, 랜드어그먼트(RandAugment)는 소수의 제어 매개변수(Control Parameter)를 통해 정책 선택(Policy Selection)을 단순화한다. 이러한 접근법은 성능을 향상할 수 있지만 계산 요구량(Computational Requirement)과 효과는 데이터셋, 모델 구조(Model Architecture), 학습 구성(Training Configuration)에 따라 달라진다.

데이터 증강은 컴퓨터 비전(Computer Vision)을 넘어 자연어 처리(Natural Language Processing), 음성 인식(Speech Recognition), 시계열 학습(Time-Series Learning)에도 적용된다. 텍스트 증강(Text Augmentation)은 제어된 단어 교체(Word Replacement)나 바꿔 쓰기(Paraphrasing)를 사용할 수 있으며, 오디오 증강(Audio Augmentation)은 배경 잡음(Background Noise), 시간적 특성(Timing), 주파수 특성(Frequency Characteristic)을 변경할 수 있다. 시계열 증강은 작업과 관련된 관계(Task-Relevant Relationship)를 유지하는 조건에서 크기 조정(Scaling), 미세 변동(Jitter), 시간 이동(Temporal Shift)을 도입할 수 있다.

로보틱스(Robotics)와 자율 시스템(Autonomous System)에서는 데이터 증강이 실제 운영 환경의 변화(Operational Variation)를 반영해야 한다. 카메라 이미지(Camera Image)는 조명 변화(Illumination Change), 기상 효과(Weather Effect), 센서 잡음, 부분 가림(Partial Occlusion)을 모사하도록 수정할 수 있다. 그러나 변환은 물리적 제약(Physical Constraint)을 준수하고 동기화된 센서(Synchronized Sensor) 사이의 기하학적 일관성(Geometric Consistency)을 유지해야 한다. 비현실적인 증강은 학습 조건과 실제 배포 환경 사이에 불일치(Mismatch)를 발생시킬 수 있다.

합성 데이터 생성(Synthetic Data Generation)과 데이터 증강은 관련되어 있지만 서로 다른 개념이다. 전통적인 데이터 증강은 기존 관측 데이터를 수정하는 반면, 합성 데이터 생성은 완전히 새로운 장면(Scene), 객체(Object), 환경 구성(Environmental Configuration)을 만들 수 있다. 시뮬레이션 플랫폼(Simulation Platform)은 제어된 조건에서 레이블이 포함된 데이터(Labeled Data)를 생성할 수 있으며, 데이터 증강은 기존 샘플의 변형 범위를 확대한다. 시뮬레이션 충실도(Simulation Fidelity)와 데이터 품질(Data Quality)이 충분하다면 두 접근법을 결합하여 데이터 범위(Coverage)를 개선할 수 있다.

데이터 증강은 드롭아웃(Dropout), 가중치 감쇠(Weight Decay), 조기 종료(Early Stopping) 등 다른 정규화 기법과 상호작용한다. 각 방법은 서로 다른 방식으로 학습에 영향을 미치므로 결합하면 일반화 성능을 향상할 수 있지만 과도한 정규화(Excessive Regularization)가 발생할 수도 있다. 예를 들어 공격적인 이미지 변환(Aggressive Image Transformation)과 강한 드롭아웃을 함께 적용하면 최적화(Optimization)가 불필요하게 어려워질 수 있다. 효과적인 조합을 결정하려면 통제된 실험(Controlled Experiment)이 필요하다.

실용적인 평가(Practical Evaluation)에서는 최소한의 전처리(Minimal Preprocessing)로 학습한 기준 모델(Baseline Model)과 점진적으로 풍부한 데이터 증강 파이프라인을 적용한 모델을 비교한다. 학습 및 검증 정확도(Training and Validation Accuracy), 손실 곡선(Loss Curve), 제어된 교란에 대한 성능은 유용한 근거를 제공한다. 비교 과정에서는 데이터 분할(Data Split), 최적화 알고리즘 설정(Optimizer Setting), 학습 예산(Training Budget)을 동일하게 유지하여 관측된 차이가 데이터 증강 선택에 기인한다고 더 신뢰성 있게 판단할 수 있도록 해야 한다.

데이터 증강 실험에서는 클래스 균형(Class Balance)과 희귀 사건 범위(Rare-Event Coverage)도 검토해야 한다. 일반적인 샘플(Common Example)을 반복적으로 변환한다고 해서 과소 대표된 클래스(Underrepresented Class)의 인식 성능이 반드시 개선되는 것은 아니다. 클래스 인식 샘플링(Class-Aware Sampling)이나 표적 데이터 증강(Targeted Augmentation)이 도움이 될 수 있지만, 희귀 샘플을 과도하게 재사용하면 인위적인 패턴(Artificial Pattern)이 발생할 수 있다. 신뢰할 수 있는 모델 평가를 위해서는 대표적인 클래스 분포(Class Distribution)와 현실적인 샘플 다양성(Sample Diversity)을 유지하는 것이 필수적이다.

데이터 증강은 실제 데이터 수집(Real-World Data Collection)에 높은 비용, 긴 시간, 운영 안전 제약(Operational Safety Constraint)이 수반되는 경우 특히 유용하다. 산업용 로보틱스(Industrial Robotics), 의료 영상(Medical Imaging), 자율 주행 및 항법(Autonomous Navigation)은 이러한 제약에 자주 직면한다. 신중하게 설계된 변환은 데이터 효율성(Data Efficiency)과 강건성을 개선할 수 있지만, 그 효과는 증강된 샘플이 실제 배포 모델이 경험할 의미 있는 변화를 얼마나 적절하게 근사하는지에 달려 있다.

데이터 증강의 효과는 궁극적으로 레이블 유효성(Label Validity)을 보존하고 현실적인 변환을 선택하며 데이터와 주석 전반에 걸쳐 일관된 구현을 유지하는 데 달려 있다. 잘 설계된 데이터 증강은 학습 중 표현되는 조건의 범위를 확대하고 모델이 전이 가능한 특징(Transferable Feature)을 학습하도록 유도한다. 데이터 증강 파이프라인, 변환 확률(Transformation Probability), 주석 처리(Annotation Handling), 평가 절차(Evaluation Procedure)를 이해하는 것은 강건한 기계학습 개발(Robust Machine Learning Development)을 위한 실용적인 기초가 된다.

## 03.04 Early Stopping

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

조기 종료(Early Stopping)는 검증 성능(Validation Performance)의 개선이 중단되었을 때 모델 학습(Model Training)을 종료하여 일반화(Generalization)를 향상하는 실용적인 정규화 기법(Regularization Technique)이다. 학습 손실(Training Loss)이 최솟값에 도달할 때까지 최적화(Optimization)를 계속하는 대신, 조기 종료는 학습에 사용되지 않은 검증 샘플(Validation Sample)에 대한 성능을 모니터링한다. 이러한 접근법은 학습 데이터셋(Training Dataset)의 잡음(Noise)과 불필요한 세부 특성에 과도하게 적합할 위험을 줄인다.

조기 종료의 필요성은 최적화 진행(Optimization Progress)과 일반화 성능 사이의 차이에서 비롯된다. 학습 과정에서 경험적 손실(Empirical Loss)은 지속적으로 감소하는 경우가 많지만, 검증 손실(Validation Loss)은 일정 시점 이후 증가할 수 있다. 이러한 차이는 추가적인 매개변수 갱신(Parameter Update)이 학습 샘플에 대한 적합도(Fitting Accuracy)는 개선하지만, 새로운 관측 데이터(New Observation)에 대한 예측 성능을 반드시 향상하지는 않는다는 것을 의미한다.

조기 종료는 학습 지속 시간(Training Duration)을 모델 복잡도 제어(Model Complexity Control)의 한 형태로 취급한다. 적은 반복 횟수(Iteration)로 학습한 신경망(Neural Network)은 데이터의 신뢰성이 낮은 변동에 적응하기 전에 주요 패턴(Dominant Pattern)을 학습할 수 있다. 최적화를 계속하면 학습 데이터에만 존재하는 특성(Training-Specific Characteristic)에 대한 민감도가 증가할 수 있다. 따라서 적절한 시점에서 학습을 중단하면 학습된 해(Learned Solution)에 암묵적인 제약(Implicit Constraint)을 부여할 수 있다.

일반적인 조기 종료 절차(Early Stopping Procedure)는 각 에포크(Epoch)가 끝나거나 지정된 학습 단계(Training Step)에 도달할 때마다 별도의 검증 데이터셋(Validation Dataset)으로 모델을 평가한다. 모니터링 지표(Monitored Metric)는 검증 손실, 분류 정확도(Classification Accuracy), F1 점수(F1 Score), 기타 작업별 평가 지표(Task-Specific Metric)가 될 수 있다. 사전에 정의한 기간 동안 지표가 개선되지 않으면 학습을 종료하고 가장 우수한 성능을 보인 모델 매개변수(Model Parameter)를 복원한다.

인내 횟수(Patience Parameter)는 충분한 성능 개선 없이 연속적으로 허용되는 검증 횟수를 결정한다. 예를 들어 patience = 10으로 설정하면 검증 평가(Validation Evaluation)에서 열 번 연속으로 개선이 나타나지 않을 때까지 학습을 계속한다. 이러한 메커니즘은 한 번의 불리한 변동(Unfavorable Fluctuation)으로 학습이 종료되는 것을 방지하며, 최적화 알고리즘(Optimizer)이 일시적인 성능 정체 구간(Performance Plateau)을 벗어날 기회를 제공한다.

최소 개선량(Minimum Improvement)은 일반적으로 min_delta로 표현되며, 의미 있는 변화로 인정할 최소한의 지표 차이를 지정한다. 검증 손실의 감소량이 이 임계값(Threshold)보다 작으면 인내 횟수 카운터(Patience Counter)가 초기화되지 않을 수 있다. 적절한 임계값을 선택하면 특히 수렴(Convergence)에 가까워져 검증 지표가 천천히 변화할 때 실제 개선(Genuine Improvement)과 수치적 변동(Numerical Fluctuation)을 구분하는 데 도움이 된다.

조기 종료를 구현하려면 모니터링 지표를 최소화(Minimization)할지 최대화(Maximization)할지 정의해야 한다. 검증 손실과 오류율(Error Rate)은 일반적으로 최소화하지만, 정확도(Accuracy), F1 점수, 여러 객체 감지 지표(Object Detection Metric)는 최대화한다. 구현 과정에서는 올바른 비교 규칙(Comparison Rule)을 적용해야 하며, 방향을 반대로 설정하면 학습이 너무 일찍 종료되거나 종료되지 않는 문제가 발생할 수 있다.

검증 빈도(Validation Frequency)는 종료 판단의 정확도(Stopping Accuracy)와 계산 오버헤드(Computational Overhead)에 영향을 준다. 매 에포크 이후 평가하면 정기적인 피드백(Feedback)을 얻을 수 있지만 대규모 데이터셋에서는 비용이 높을 수 있다. 단계 기반 검증(Step-Based Validation)은 에포크가 긴 경우 더 세밀한 제어를 제공하지만, 빈번한 평가는 전체 학습 시간을 증가시킬 수 있다. 종료 설정(Stopping Configuration)은 계산 예산(Computational Budget)과 예상되는 모델 개선 속도를 반영해야 한다.

기본적인 파이썬(Python) 구현은 최상의 검증 점수(Best Validation Score), 개선되지 않은 평가 횟수를 저장하는 카운터(Counter), 인내 횟수 임계값(Patience Threshold)을 유지한다. 각 검증 단계 이후 현재 점수를 기록된 최상의 결과와 비교한다. 의미 있는 개선이 발생하면 카운터를 초기화하고, 개선이 충분하지 않으면 카운터를 증가시킨다. 카운터가 설정된 인내 횟수에 도달하면 학습을 종료한다.

최상의 모델 체크포인트(Best Model Checkpoint)를 저장하는 것은 필수적이다. 마지막 학습 반복(Training Iteration)이 반드시 가장 우수한 검증 성능을 나타내는 시점은 아니기 때문이다. 체크포인트에는 모델 가중치(Model Weight), 최적화 알고리즘 상태(Optimizer State), 에포크 번호(Epoch Number), 선택된 학습 메타데이터(Training Metadata)가 포함될 수 있다. 최상의 가중치를 복원하면 이후 평가나 배포(Deployment)에서 선택된 검증 결과에 해당하는 모델을 사용할 수 있다.

파이토치(PyTorch)에서는 일반적으로 학습 루프(Training Loop) 내부에 사용자 정의 유틸리티(Custom Utility)를 구현하여 조기 종료를 처리한다. 에포크가 완료되면 model.eval()을 통해 모델을 평가 모드(Evaluation Mode)로 전환하고, 기울기 추적(Gradient Tracking) 없이 검증 예측(Validation Prediction)을 계산한다. 이후 검증 지표를 조기 종료 제어기(Early Stopping Controller)에 전달하여 체크포인트 저장, 학습 지속, 학습 종료 여부를 결정한다.

드롭아웃(Dropout)이나 배치 정규화(Batch Normalization)를 포함하는 모델을 평가할 때는 올바른 모드 관리(Mode Management)가 중요하다. model.eval()을 호출하면 표준 드롭아웃(Standard Dropout)이 비활성화되고 배치 정규화는 저장된 통계량(Stored Statistics)을 사용하도록 동작이 변경된다. 검증 계산에서는 일반적으로 torch.no_grad() 또는 torch.inference_mode()를 사용하여 불필요한 기울기 추적을 방지하고 메모리 사용량(Memory Consumption)을 줄여야 한다.

검증 이후 학습을 재개할 때는 model.train()을 호출하여 학습 동작(Training Behavior)을 복원해야 한다. 그렇지 않으면 드롭아웃과 정규화 계층(Normalization Layer)이 이후 최적화 단계에서 잘못 작동할 수 있다. 신뢰할 수 있는 구현은 학습 단계(Training Phase)와 평가 단계(Evaluation Phase)를 명확하게 분리하여 검증 결과의 일관성을 유지하고 적절한 학습 동역학(Training Dynamics)을 보장한다.

재사용 가능한 조기 종료 클래스(EarlyStopping Class)는 인내 횟수, 최소 개선량, 최상의 지표(Best Metric), 연속적인 성능 미개선 횟수를 저장할 수 있다. 갱신 메서드(Update Method)는 최신 검증 결과를 입력받아 종료 여부를 반환한다. 선택적인 체크포인트 기능(Checkpoint Functionality)은 성능이 개선될 때마다 모델 매개변수를 저장할 수 있으며, 이를 통해 특정 신경망 구조(Network Architecture)에 종속되지 않는 구성 요소를 만들 수 있다.

체크포인트 복원(Checkpoint Restoration)에서는 변경 가능한 매개변수 객체(Mutable Parameter Object)에 대한 참조(Reference)가 최적화가 진행되면서 달라질 수 있으므로 주의해야 한다. 메모리 내 체크포인트(In-Memory Checkpoint)는 모델 상태(Model State)의 독립적인 복사본을 사용해야 하며, 디스크 기반 체크포인트(Disk-Based Checkpoint)는 상태 사전(State Dictionary)을 직렬화(Serialization)할 수 있다. 추론만 필요하다면 모델 가중치 복원으로 충분할 수 있지만, 정확한 학습 재개(Training Continuation)를 위해서는 최적화 알고리즘과 스케줄러 상태(Scheduler State)도 필요하다.

조기 종료는 편향-분산 절충 관계(Bias-Variance Tradeoff)와 밀접하게 관련된다. 학습을 지나치게 일찍 종료하면 모델이 충분한 구조를 학습하지 못해 과소적합(Underfitting)이 발생할 수 있다. 반대로 학습을 너무 오래 지속하면 잡음과 데이터셋 고유 패턴(Dataset-Specific Pattern)에 대한 민감도가 증가할 수 있다. 종료 시점(Stopping Point)은 실용적인 균형을 추구하지만, 학습 지속 시간과 일반화 사이의 관계가 항상 단조적(Monotonic)인 것은 아니다.

이 기법은 검증 성능이 안정화된 이후에도 학습 손실을 계속 감소시킬 수 있는 과매개변수화 신경망(Overparameterized Neural Network)에 특히 유용하다. 그러나 검증 곡선(Validation Curve)은 일시적인 성능 저하 이후 다시 개선되는 형태를 나타낼 수 있다. 따라서 조기 종료에서는 검증 손실이 단기간 증가했다고 해서 반드시 회복할 수 없는 과적합(Irreversible Overfitting)이 발생했다고 가정해서는 안 된다.

학습률 스케줄(Learning Rate Schedule)은 조기 종료와 강하게 상호작용할 수 있다. 예정된 학습률 감소(Learning Rate Reduction)가 추가적인 개선을 가능하게 하기 전까지 모델의 검증 성능이 거의 변하지 않을 수 있다. 인내 횟수가 너무 짧으면 이러한 개선이 발생하기 전에 학습이 종료될 수 있다. 종료 기준(Stopping Criterion)을 학습률 스케줄러(Learning Rate Scheduler)와 조정하면 계획된 최적화 전환(Optimization Transition) 과정에서 조기 종료가 너무 빨리 발생하는 것을 방지할 수 있다.

조기 종료는 모니터링 지표의 개선이 중단되면 학습률을 낮추는 ReduceLROnPlateau와 결합할 수도 있다. 이러한 구성에서는 일반적으로 조기 종료가 학습을 중단하기 전에 학습률 스케줄러가 최적화 과정을 조정할 기회를 제공한다. 모델이 학습률 감소의 효과를 충분히 활용할 수 있도록 두 기능의 인내 횟수 설정(Patience Setting)을 조정해야 한다.

검증 데이터(Validation Data)는 의미 있는 종료 신호(Stopping Signal)를 제공하기 위해 학습 데이터(Training Data)와 분리되어야 한다. 동일한 샘플을 최적화와 종료 판단에 모두 사용하면 모니터링 지표가 과적합을 제대로 드러내지 못할 수 있다. 전처리 절차(Preprocessing Procedure)에서도 정보 누출(Information Leakage)을 방지해야 하며, 특히 정규화 통계량(Normalization Statistics)이나 특징 변환(Feature Transformation)을 데이터로부터 추정할 때 주의가 필요하다.

동일한 검증 데이터셋을 반복적으로 사용하여 체크포인트를 선택하면 일종의 모델 선택 편향(Model Selection Bias)이 발생한다. 조기 종료는 많은 경우 일반화를 개선하지만, 광범위한 튜닝(Extensive Tuning)이 수행되면 검증 성능이 실제보다 낙관적으로 평가될 수 있다. 따라서 여러 모델 구조나 종료 설정을 비교할 때는 최종 평가(Final Evaluation)를 위한 별도의 테스트 데이터셋(Test Dataset)을 확보해야 한다.

검증 데이터셋이 작으면 일부 샘플이 결과에 큰 영향을 미치므로 지표 추정치(Metric Estimate)가 상당히 변동할 수 있다. 인내 횟수를 늘리거나 더 안정적인 평가 지표(Stable Evaluation Metric)를 사용하면 너무 이른 종료를 줄일 수 있다. 그러나 지나치게 긴 인내 횟수는 정규화 효과를 약화시킬 수 있다. 적절한 설정은 데이터셋 크기, 측정 잡음(Measurement Noise), 조기 종료로 인한 영향을 고려해야 한다.

조기 종료는 불필요한 학습 에포크를 줄여 계산 비용(Computational Cost)을 절감할 수 있다. 이러한 이점은 고가의 그래픽 처리 장치(GPU) 자원이나 장시간 분산 학습(Distributed Training)이 필요한 모델에서 더욱 중요하다. 그러나 빈번한 검증과 체크포인트 직렬화(Checkpoint Serialization)는 추가적인 오버헤드를 발생시킨다. 효율적인 구현에서는 학습을 일찍 종료하여 절약하는 비용과 모델 성능 모니터링 비용 사이의 균형을 고려해야 한다.

분산 학습에서는 종료 여부를 결정하기 전에 여러 작업자(Worker)의 검증 지표를 집계(Aggregation)해야 할 수 있다. 모든 작업자가 일관되게 종료되도록 결정 결과를 동기화(Synchronization)해야 한다. 중복 파일(Duplicate File)이나 경쟁 상태(Race Condition)를 방지하기 위해 체크포인트 기록(Checkpoint Writing)은 일반적으로 지정된 프로세스(Process)에서만 수행한다. 이러한 고려 사항은 신뢰할 수 있는 다중 GPU 학습 파이프라인(Multi-GPU Training Pipeline)을 구축하는 데 중요하다.

실용적인 실험(Practical Experiment)에서는 다른 조건을 동일하게 유지하면서 고정 기간 학습(Fixed-Duration Training)과 조기 종료를 비교한다. 학습 및 검증 손실 곡선(Training and Validation Loss Curve)은 종료 기준이 일반화 동작(Generalization Behavior)에 어떻게 반응하는지 보여준다. 이후 선택된 체크포인트를 독립적인 테스트 데이터셋(Independent Test Set)에서 평가할 수 있다. 최상의 에포크(Best Epoch), 종료 에포크(Stopping Epoch), 인내 횟수 카운터를 기록하면 실험 재현성(Experiment Reproducibility)을 향상할 수 있다.

조기 종료는 분류(Classification)뿐 아니라 회귀(Regression), 객체 감지(Object Detection), 분할(Segmentation), 시퀀스 모델링(Sequence Modeling)에도 적용할 수 있다. 각 응용 분야에서는 실제 배포 목적(Deployment Objective)을 반영하는 검증 지표를 선택해야 한다. 객체 감지에서는 검증 손실이 평균 정밀도(Mean Average Precision, mAP)와 항상 완벽하게 일치하지 않을 수 있으며, 회귀 시스템에서는 평균 절대 오차(Mean Absolute Error, MAE) 또는 다른 응용 분야별 지표(Application-Specific Metric)를 우선할 수 있다.

조기 종료는 신뢰할 수 있는 검증 절차(Reliable Validation Procedure), 적절한 인내 횟수 설정, 의미 있는 최소 개선 임계값(Minimum Improvement Threshold), 체크포인트 복원을 결합할 때 가장 효과적이다. 최적의 일반화(Optimal Generalization)를 보장하지는 않지만, 학습 지속 시간을 제어하고 학습 데이터에 대한 불필요한 적응(Unnecessary Adaptation)을 제한하는 효율적인 메커니즘을 제공한다. 구현 원리를 이해하면 더욱 신뢰할 수 있고 계산 효율적인 기계학습 개발(Machine Learning Development)이 가능해진다.

## 03.05 Label Smoothing

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

레이블 스무딩(Label Smoothing)은 예측된 클래스 확률(Class Probability)에 대한 과도한 확신(Overconfidence)을 줄이기 위해 분류 모델(Classification Model)에 사용하는 정규화 기법(Regularization Technique)이다. 기존 지도학습(Supervised Learning)은 일반적으로 정답 클래스(Correct Class)에 확률 1을, 나머지 모든 클래스에 확률 0을 할당하는 원-핫 벡터(One-Hot Vector)로 목표값(Target)을 표현한다. 레이블 스무딩은 이러한 극단적인 목표값을 완화된 확률 분포(Softened Probability Distribution)로 대체하여 보다 균형 잡힌 예측을 유도하고 일반화(Generalization)를 개선할 수 있다.

레이블 스무딩의 필요성은 모든 학습 레이블(Training Label)을 절대적으로 확실한 관측 결과로 취급하는 방식의 한계에서 비롯된다. 실제 데이터셋(Dataset)에는 주석 오류(Annotation Error), 모호한 샘플(Ambiguous Example), 중첩된 클래스 경계(Overlapping Class Boundary)가 포함될 수 있다. 모든 레이블 샘플에 대해 거의 완벽한 확신도를 출력하도록 학습하면 지나치게 날카로운 결정 경계(Decision Boundary)와 부정확하게 보정된 예측(Poorly Calibrated Prediction)이 발생할 수 있으며, 특히 학습 데이터가 제한적이거나 잡음이 많은 경우 이러한 문제가 두드러진다.

K개의 클래스로 구성된 분류 문제(Classification Problem)에서 레이블 스무딩은 평활화 계수(Smoothing Coefficient) ε을 사용하여 원래의 원-핫 목표 분포(One-Hot Target Distribution)를 수정한다. 일반적인 수식은 qᵢ = (1 − ε)yᵢ + ε/K이며, 여기서 yᵢ는 원래 목표값이고 qᵢ는 평활화된 목표 확률(Smoothed Target Probability)이다. 정답 클래스에는 1 − ε + ε/K가 할당되고, 각 오답 클래스(Incorrect Class)에는 ε/K가 할당되어 모든 목표 확률의 합이 1이 되도록 한다.

평활화 계수는 목표 분포(Target Distribution)가 균등 분포(Uniform Distribution)에 얼마나 가까워지는지를 결정한다. ε = 0이면 기존 원-핫 레이블(One-Hot Label)이 변경되지 않는다. ε이 증가하면 정답 클래스의 목표 확률은 감소하고 다른 클래스에 할당되는 확률은 증가한다. 지나치게 강한 평활화(Excessive Smoothing)는 유용한 클래스 정보(Class Information)를 제거할 수 있으므로 데이터셋과 학습 목적(Learning Objective)에 따라 계수를 선택해야 한다.

레이블 스무딩은 신경망 분류(Neural Network Classification)에 사용되는 교차 엔트로피 손실(Cross-Entropy Loss)에 일반적으로 적용된다. 표준 교차 엔트로피는 L = −Σᵢ yᵢ log(pᵢ)로 표현되며, 여기서 pᵢ는 클래스 i의 예측 확률(Predicted Probability)이다. 레이블 스무딩을 적용하면 목표 분포가 q로 변경되어 L = −Σᵢ qᵢ log(pᵢ)가 된다. 이러한 수정은 모델 구조(Model Architecture)를 변경하지 않고도 최적화(Optimization)를 유도하는 기울기(Gradient)를 변화시킨다.

기울기 동작(Gradient Behavior)은 레이블 스무딩이 과도한 확신을 줄이는 이유를 설명한다. 소프트맥스 분류(Softmax Classification)와 교차 엔트로피 손실에서 각 로짓(Logit)에 대한 기울기는 pᵢ − qᵢ에 비례한다. 원-핫 목표값과 달리 평활화된 레이블(Smoothed Label)은 정답 클래스 확률이 1에 접근하도록 강하게 유도하지 않는다. 따라서 최적화 알고리즘(Optimizer)은 이미 지배적인 로짓(Dominant Logit)을 무한정 증가시킬 유인이 줄어든다.

레이블 스무딩은 모델 매개변수(Model Parameter)에 직접 벌점(Penalty)을 부과하지 않고 학습 목표값(Training Target)을 수정한다는 점에서 L1 및 L2 정규화(Regularization)와 다르다. L1은 희소 계수(Sparse Coefficient)를 유도하고, L2는 큰 매개변수 크기(Parameter Magnitude)를 억제한다. 반면 레이블 스무딩은 분류기가 학습하는 확률 분포(Probability Distribution)에 영향을 준다. 정규화 효과는 완화된 목표값, 예측 확신도(Predictive Confidence), 최적화 동역학(Optimization Dynamics) 사이의 관계에서 발생한다.

확률론적 해석(Probabilistic Interpretation)에서는 레이블 스무딩을 관측된 레이블 분포(Observed Label Distribution)와 클래스에 대한 사전 분포(Prior Distribution)를 결합하는 방식으로 이해한다. 균등 레이블 스무딩(Uniform Label Smoothing)은 목표 확률의 일부를 모든 클래스에 동일하게 분배해야 한다고 가정한다. 이러한 가정은 레이블에 불확실성(Uncertainty)이 존재할 때 유용할 수 있지만, 균등 분포가 의미적으로 관련된 클래스(Semantically Related Class) 사이의 실제 모호성을 정확하게 표현하지는 못할 수 있다.

레이블 스무딩은 예측 엔트로피(Prediction Entropy)와의 관계를 통해서도 이해할 수 있다. 확신도가 높은 분류기(Highly Confident Classifier)는 하나의 클래스에 확률이 집중된 낮은 엔트로피(Low-Entropy)의 확률 분포를 출력하는 경우가 많다. 평활화된 목표값은 이러한 극단적인 집중을 억제하며 학습 중 예측 엔트로피(Predictive Entropy)를 증가시킬 수 있다. 그러나 높은 엔트로피가 자동으로 더 나은 불확실성 추정(Uncertainty Estimation)을 의미하지는 않으며, 확률 보정(Calibration)에 미치는 영향은 실험적으로 평가해야 한다.

실제 구현(Practical Implementation)에서는 손실(Loss)을 계산하기 전에 완화된 목표 벡터(Softened Target Vector)를 생성하여 레이블 스무딩을 적용할 수 있다. 넘파이(NumPy) 구현에서는 먼저 각 클래스 레이블(Class Label)을 원-핫 표현(One-Hot Representation)으로 변환하고, 여기에 1 − ε을 곱한 다음 모든 클래스 성분(Class Component)에 ε/K를 더한다. 이렇게 생성된 확률 벡터(Probability Vector)는 교차 엔트로피를 사용하여 학습 목적 함수(Training Objective)를 계산하는 데 활용된다.

구현 과정에서는 목표 확률(Target Probability)이 음수가 아니며 합계가 1인지 확인해야 한다. 균등 평활화(Uniform Smoothing)에서는 일반적으로 ε을 0에서 1 사이의 구간에서 선택한다. 0에 가까운 확률의 로그(Logarithm)를 직접 계산하면 불안정한 값이 발생할 수 있으므로 수치적 안정성(Numerical Stability)도 중요하다. 최신 프레임워크(Framework)는 일반적으로 로그 소프트맥스(Log-Softmax) 연산이나 안정적인 교차 엔트로피 구현을 사용한다.

파이토치(PyTorch)는 torch.nn.CrossEntropyLoss의 label_smoothing 인자를 통해 내장된 레이블 스무딩(Built-In Label Smoothing)을 제공한다. 예를 들어 nn.CrossEntropyLoss(label_smoothing=0.1)은 손실 계산 과정에서 목표 분포에 균등 평활화를 적용한다. 이러한 접근법은 밀집 목표 벡터(Dense Target Vector)를 수동으로 생성할 필요가 없으며 일반적인 분류 학습 루프(Classification Training Loop)에 직접 통합할 수 있다.

일반적인 파이토치 구현에서는 신경망(Neural Network)을 통해 로짓을 계산하고, 로짓과 정수형 클래스 레이블(Integer Class Label)을 설정된 손실 함수(Loss Function)에 전달한 다음 역전파(Backward Propagation)를 수행한다. 평활화 연산(Smoothing Operation)은 손실 함수 내부에서 처리된다. 추론(Inference) 과정에서는 모델이 평소와 동일하게 로짓이나 확률을 출력하는데, 레이블 스무딩은 별도의 추론 연산을 추가하는 것이 아니라 학습 목적 함수를 수정하기 때문이다.

텐서플로(TensorFlow)와 케라스(Keras)도 적절한 교차 엔트로피 손실 설정(Cross-Entropy Loss Configuration)을 통해 레이블 스무딩을 지원한다. 범주형 분류(Categorical Classification)에서는 tf.keras.losses.CategoricalCrossentropy를 사용하여 원-핫 목표값이나 확률 목표값(Probability Target)에 평활화를 적용할 수 있다. 손실 함수 구현에 따라 사용 가능한 평활화 옵션(Smoothing Option)과 예상 목표 형식(Target Format)이 다르므로 범주형 목표값과 희소 정수 레이블(Sparse Integer Label)을 구분해야 한다.

레이블 스무딩은 대규모 신경망(Large Neural Network)이 학습 샘플에 대해 지나치게 높은 확신도를 나타낼 수 있는 이미지 분류(Image Classification)에서 특히 유용하다. 하나의 클래스에 거의 모든 확률 질량(Probability Mass)을 할당하도록 요구하는 압력을 줄이면 적절한 조건에서 일반화를 개선할 수 있다. 그러나 효과는 데이터셋 품질(Dataset Quality), 모델 용량(Model Capacity), 데이터 증강 정책(Data Augmentation Policy), 다른 정규화 메커니즘(Regularization Mechanism)의 존재 여부에 따라 달라진다.

객체 감지(Object Detection)에서는 객체 범주(Object Category)를 예측하는 분류 구성 요소(Classification Component)에 레이블 스무딩을 적용할 수 있다. 객체 감지 모델 구조(Detection Architecture)는 일반적으로 분류, 위치 추정(Localization), 객체 존재성(Objectness)에 대한 목적 함수(Objective)를 결합하므로 적절한 손실 항(Loss Term)에 평활화를 통합해야 한다. 회귀 목표값(Regression Target)이나 위치 좌표(Localization Coordinate)에 무분별하게 평활화를 적용하는 것은 이러한 출력이 범주형 확률 분포(Categorical Probability Distribution)를 나타내지 않으므로 개념적으로 잘못된 접근이다.

의미론적 분할(Semantic Segmentation)에서도 각 픽셀(Pixel)에 범주형 클래스 분포(Categorical Class Distribution)가 할당될 수 있으므로 레이블 스무딩의 이점을 얻을 수 있다. 그러나 분할 데이터셋(Segmentation Dataset)에는 무시 대상 픽셀(Ignored Pixel), 불확실한 경계(Uncertain Boundary), 심각한 클래스 불균형(Class Imbalance)이 자주 포함된다. 구현에서는 무시 마스크(Ignore Mask)를 보존하고 유효하지 않은 클래스(Invalid Class)에 의도치 않게 확률 질량을 할당하지 않도록 해야 한다. 지나친 평활화는 세밀한 객체 경계(Fine Object Boundary)를 구분하는 능력을 저하시킬 수 있다.

레이블 스무딩과 지식 증류(Knowledge Distillation)의 관계는 현대 신경망 학습에서 중요하다. 지식 증류는 교사 모델(Teacher Model)의 부드러운 예측(Soft Prediction)을 사용하여 학생 모델(Student Model)을 지도하지만, 일반적인 레이블 스무딩은 사전에 정의된 규칙에 따라 확률 질량을 분배한다. 두 기법 모두 감독 신호(Supervision)를 완화하지만, 교사 모델이 생성한 목표값은 균등 평활화가 표현하지 못하는 클래스 간 관계(Class Relationship)를 포함할 수 있다.

레이블 스무딩은 표현 학습(Representation Learning)과 학습된 특징(Learned Feature)의 구조에도 영향을 줄 수 있다. 모든 레이블 샘플과 대체 클래스(Alternative Class) 사이의 분리를 극대화하려는 유인을 줄임으로써 특징 임베딩(Feature Embedding)이 구성되는 방식을 변화시킬 수 있다. 이러한 효과는 분류 성능(Classification Performance)을 향상할 수 있지만, 일부 지식 증류 작업을 포함한 특정 후속 작업(Downstream Task)에 유용한 정보를 감소시킬 수도 있다.

확률 보정(Calibration)도 중요한 고려 사항이다. 보정된 분류기(Calibrated Classifier)는 예측 확률이 실제 관측된 정답 빈도(Observed Correctness Frequency)와 합리적으로 일치하도록 출력해야 한다. 레이블 스무딩은 극단적인 확신도를 줄이고 경우에 따라 보정 지표(Calibration Metric)를 개선할 수 있지만, 확률 보정을 보장하지는 않는다. 신뢰도 다이어그램(Reliability Diagram), 기대 보정 오차(Expected Calibration Error), 음의 로그 우도(Negative Log-Likelihood) 및 관련 지표를 사용하여 실제 효과를 평가할 수 있다.

평활화 계수는 중요한 하이퍼파라미터(Hyperparameter)이므로 검증 데이터(Validation Data)를 사용하여 조정해야 한다. 0.05나 0.1과 같은 작은 값은 초기 실험값(Starting Point)으로 자주 검토되지만 모든 문제에서 최적인 것은 아니다. 더 강한 평활화는 잡음이 많은 일부 분류 문제(Noisy Classification Problem)에 도움이 될 수 있지만, 클래스를 명확하게 구분해야 하는 작업에서는 성능을 저하시킬 수 있다. 적절한 설정을 찾기 위해서는 통제된 실험(Controlled Experiment)이 필요하다.

레이블 스무딩은 데이터 증강(Data Augmentation), 드롭아웃(Dropout), 가중치 감쇠(Weight Decay), 기타 정규화 방법(Regularization Method)과 상호작용한다. 이러한 기법을 결합하면 강건성(Robustness)을 향상할 수 있지만 과도한 정규화(Excessive Regularization)는 모델이 중요한 패턴을 학습하지 못하도록 할 수 있다. 강한 데이터 증강(Strong Augmentation)이 이미 학습 난이도(Training Difficulty)를 상당히 높이고 있다면 추가적인 평활화는 이점이 제한적이거나 과소적합(Underfitting)을 증가시킬 수 있다.

클래스 불균형은 균등 레이블 스무딩의 해석을 복잡하게 만든다. 모든 클래스에 동일한 확률 질량을 분배하는 방식은 불균등한 클래스 빈도(Class Frequency)나 비대칭 분류 비용(Asymmetric Classification Cost)을 반영하지 않는다. 일부 응용 분야에서는 클래스별 평활화(Class-Dependent Smoothing)나 대체 목표 분포(Alternative Target Distribution)가 더 적절할 수 있다. 이러한 수정은 자동으로 적용하기보다 학습 목적에 근거하여 정당화해야 한다.

잡음 레이블 학습(Noisy-Label Learning)에서는 레이블 스무딩이 잘못된 주석(Incorrect Annotation)을 수정하지 않는다는 점에 주의해야 한다. 잘못 레이블링된 샘플(Mislabeled Example)은 평활화 이후에도 기록된 클래스(Recorded Class)에 가장 높은 목표 확률을 할당한다. 확신도를 높이려는 압력이 감소하면서 일부 부정적인 영향이 완화될 수 있지만, 체계적인 주석 오류(Systematic Annotation Error)는 레이블 정제(Label Cleaning), 강건한 손실 함수(Robust Loss Function), 전용 잡음 처리 전략(Noise-Handling Strategy)을 요구할 수 있다.

실용적인 평가(Practical Evaluation)에서는 일반적인 교차 엔트로피 기준 모델(Cross-Entropy Baseline)과 여러 평활화 계수를 적용하여 학습한 모델을 비교한다. 학습 손실(Training Loss), 검증 정확도(Validation Accuracy), 음의 로그 우도, 확률 보정 지표는 상호 보완적인 근거를 제공한다. 평활화 적용 여부에 따라 서로 다른 목표값을 최적화하므로, 두 방식의 학습 손실 수치(Numerical Training-Loss Value)를 직접 비교할 때는 신중하게 해석해야 한다.

구현 테스트(Implementation Testing)에서는 대규모 실험을 수행하기 전에 수학적인 목표 분포(Mathematical Target Distribution)를 검증해야 한다. 예를 들어 K = 5이고 ε = 0.1이면 정답 클래스의 목표 확률은 0.92가 되고, 나머지 각 클래스에는 0.02가 할당된다. 이러한 값을 확인하면 클래스 수(Class Count), 평활화 규칙(Smoothing Convention), 잘못된 정규화(Incorrect Normalization)와 관련된 오류를 발견하는 데 도움이 된다.

레이블 스무딩은 네트워크 구조를 변경하지 않고 목표 확률이나 손실 계산만 수정하므로 계산 오버헤드(Computational Overhead)가 비교적 작다. 프레임워크 구현(Framework Implementation)에서는 전체 목표 분포(Full Target Distribution)를 명시적으로 메모리에 생성하지 않고 계산할 수도 있다. 따라서 레이블 스무딩은 대규모 분류(Large-Scale Classification)에 매력적인 기법이지만, 메모리 사용량(Memory Usage)과 효율성(Efficiency)은 구체적인 구현 방식과 클래스 수에 따라 달라진다.

레이블 스무딩은 과도한 확신을 유도하는 감독 신호(Overconfident Supervision)를 완화하고 극단적이지 않은 클래스 확률 분포(Class Probability Distribution)를 학습하도록 유도하는 간단한 메커니즘을 제공한다. 그 효과는 적절한 계수 선택(Coefficient Selection), 올바른 손실 함수 구현(Loss Implementation), 유효한 클래스 정의(Class Definition), 신뢰할 수 있는 평가(Reliable Evaluation)에 달려 있다. 수학적 공식(Mathematical Formulation)과 실제 동작(Practical Behavior)을 이해하면 완화된 학습 목표값(Softened Training Target)의 한계를 인식하면서 분류 모델의 일반화 성능을 개선할 수 있다.

## 03.06 Implicit Regularization

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

암묵적 정규화(Implicit Regularization)는 목적 함수(Objective Function)에 명시적인 벌점(Explicit Penalty)을 추가하지 않더라도 기계학습 알고리즘(Machine Learning Algorithm)이 특정 해(Solution)를 선호하는 경향을 의미한다. L1 정규화(L1 Regularization), L2 정규화(L2 Regularization), 드롭아웃(Dropout)과 달리 암묵적 정규화는 최적화 절차(Optimization Procedure) 자체에서 발생한다. 최적화 알고리즘(Optimizer), 학습률(Learning Rate), 초기화(Initialization), 학습 동역학(Training Dynamics)은 학습 데이터에 적합한 여러 모델 중 어떤 해가 선택되는지에 영향을 줄 수 있다.

암묵적 정규화의 중요성은 학습 가능한 매개변수(Trainable Parameter)의 수가 학습 샘플(Training Example)의 수를 크게 초과할 수 있는 과매개변수화 신경망(Overparameterized Neural Network)에서 특히 뚜렷하게 나타난다. 이러한 모델은 학습 오차(Training Error)를 거의 0으로 줄이면서도 효과적인 일반화(Generalization)를 달성할 수 있다. 이러한 현상은 모델 용량(Model Capacity)만으로 일반화 성능이 결정되지 않으며, 최적화 동역학이 학습된 해의 구조에 기여한다는 것을 보여준다.

일반적으로 SGD로 약칭하는 확률적 경사하강법(Stochastic Gradient Descent)은 딥러닝(Deep Learning)에서 암묵적 정규화를 발생시키는 주요 요인이다. SGD는 전체 학습 데이터셋(Training Dataset)을 사용하여 기울기(Gradient)를 계산하는 대신 무작위로 선택한 샘플이나 미니배치(Mini-Batch)에서 기울기를 추정한다. 이러한 추정은 매개변수 갱신(Parameter Update)에 확률적 변동(Stochastic Variation)을 도입하여 결정론적 전체 배치 경사하강법(Deterministic Full-Batch Gradient Descent)과 다른 최적화 경로(Optimization Trajectory)를 형성한다.

모델 매개변수 θ에 대한 일반적인 SGD 갱신은 θₜ₊₁ = θₜ − ηgₜ로 표현되며, 여기서 η는 학습률이고 gₜ는 확률적 기울기 추정치(Stochastic Gradient Estimate)이다. 기울기는 gₜ = ∇L(θₜ) + ξₜ로 분해할 수 있으며, ξₜ는 샘플링 잡음(Sampling Noise)을 나타낸다. 균등하고 편향되지 않은 샘플링(Uniform Unbiased Sampling)에서는 확률적 기울기의 기댓값(Expected Value)이 전체 경험적 기울기(Full Empirical Gradient)와 같지만, 개별 갱신은 그 주변에서 변동한다.

기울기 잡음(Gradient Noise)은 최적화 알고리즘이 손실 지형(Loss Landscape)을 탐색하는 방식에 영향을 준다. 여러 가능한 해가 존재하는 영역에서 확률적 갱신은 특정한 좁거나 불안정한 구성(Narrow or Unstable Configuration)에 장기간 머무르는 것을 억제할 수 있다. 이러한 동작은 상대적으로 낮은 손실이 넓은 영역에 걸쳐 유지되는 해를 선호하는 경향에 영향을 줄 수 있다. 그러나 기울기 잡음, 평평한 최솟값(Flat Minimum), 일반화 사이의 관계는 모델 매개변수화(Model Parameterization)와 문제 구조(Problem Structure)에 따라 달라진다.

평평한 최솟값(Flat Minima)이라는 개념은 암묵적 정규화를 설명하는 데 자주 사용된다. 평평한 최솟값은 적당한 매개변수 교란(Parameter Perturbation)이 학습 손실(Training Loss)에 비교적 작은 변화를 일으키는 영역을 비공식적으로 의미한다. 이러한 해는 급격하게 국소화된 최솟값(Sharply Localized Minimum)보다 특정 교란을 더 잘 견딜 수 있다. 그러나 수치적 평탄성(Numerical Flatness)은 모든 매개변수 변환(Parameter Transformation)에 대해 불변하지 않으며, 우수한 일반화를 보편적으로 보장하지도 않는다.

SGD는 보다 단순한 수학적 모델(Mathematical Model)에서도 암묵적 편향(Implicit Bias)을 나타낸다. 예를 들어 특정 미결정 선형 최소제곱 문제(Underdetermined Linear Least-Squares Problem)에서 경사하강법(Gradient Descent)은 0으로 초기화할 경우 최소 유클리드 노름 해(Minimum-Euclidean-Norm Solution)로 수렴한다. 로지스틱 손실(Logistic Loss)을 사용하는 선형 분리 가능 분류(Separable Linear Classification)에서는 적절한 조건에서 경사하강법이 최대 마진 방향(Maximum-Margin Direction)에 접근할 수 있다. 이러한 사례는 명시적인 정규화 항(Explicit Regularization Term) 없이도 최적화가 구조적인 해를 선택할 수 있음을 보여준다.

학습률은 수렴 동작(Convergence Behavior)과 암묵적 정규화 모두에 영향을 준다. 큰 학습률은 확률적 매개변수 이동(Stochastic Parameter Movement)의 규모를 증가시키고 일부 날카로운 구성(Sharp Configuration)으로의 수렴을 방해할 수 있다. 작은 학습률은 일반적으로 해 근처에서 더 정밀한 최적화를 지원한다. 그러나 지나치게 큰 학습률은 불안정성(Instability)이나 발산(Divergence)을 일으킬 수 있으며, 매우 작은 학습률은 상당한 계산량(Computation)을 요구할 수 있다.

학습률 스케줄링(Learning Rate Scheduling)은 학습 전반에 걸쳐 최적화 경로를 변화시킨다. 일반적인 전략은 비교적 큰 학습률로 시작한 다음 학습이 진행됨에 따라 점진적으로 감소시키는 것이다. 초기의 확률적 탐색(Stochastic Exploration) 이후 더욱 안정적인 세부 조정(Stable Refinement)이 이루어질 수 있다. 최종 모델은 마지막 학습률뿐 아니라 전체 스케줄(Complete Schedule)에 영향을 받으므로 학습률 스케줄링은 암묵적 정규화의 중요한 구성 요소가 된다.

모멘텀(Momentum)은 이전 기울기(Previous Gradient)의 정보를 누적하여 SGD를 수정한다. 일반적인 수식은 vₜ = μvₜ₋₁ + gₜ 및 θₜ₊₁ = θₜ − ηvₜ이며, 여기서 μ는 모멘텀을 제어한다. 이 메커니즘은 일부 기울기 변동을 완화하고 지속적인 방향(Persistent Direction)으로의 이동을 가속한다. 모멘텀은 최적화 알고리즘의 암묵적 선호(Implicit Preference)도 변화시키지만, 일반화에 미치는 영향은 데이터셋과 학습 구성(Training Configuration)에 따라 달라진다.

Adam과 AdamW 같은 적응형 최적화 알고리즘(Adaptive Optimizer)은 기울기 통계량(Gradient Statistics)의 추정치를 사용하여 매개변수 변화량을 조정함으로써 서로 다른 갱신 동역학(Update Dynamics)을 도입한다. 이러한 방법은 빠르게 수렴하고 서로 다른 기울기 크기(Heterogeneous Gradient Magnitude)를 효과적으로 처리할 수 있다. 그러나 암묵적 편향은 SGD와 다를 수 있다. 어느 최적화 알고리즘 계열도 항상 더 나은 일반화를 제공하는 것은 아니므로 학습률, 정규화, 계산 예산(Computational Budget)을 통제하여 비교해야 한다.

미니배치 크기(Mini-Batch Size)는 암묵적 정규화에 영향을 주는 또 다른 주요 요인이다. 미니배치는 최적화 단계(Optimization Step)에서 기울기를 추정하는 데 사용되는 학습 샘플의 부분집합(Subset)이다. 작은 배치는 변동성이 큰 기울기 추정치를 생성하는 반면, 큰 배치는 일반적으로 샘플링 잡음을 줄인다. 따라서 배치 크기(Batch Size)는 최적화 동역학, 하드웨어 활용률(Hardware Utilization), 메모리 사용량(Memory Consumption), 잠재적인 일반화 성능에 영향을 준다.

독립적인 샘플링(Independent Sampling)에서는 평균 미니배치 기울기(Averaged Mini-Batch Gradient)의 공분산(Covariance)이 일반적으로 배치 크기 B에 대해 약 1/B에 비례하여 감소한다. 유한한 데이터셋에서 비복원 추출(Sampling Without Replacement)을 수행하면 추가적인 유한 모집단 보정(Finite-Population Correction)이 적용된다. 이러한 관계는 큰 배치가 일반적으로 더 안정적인 기울기 추정치를 제공하는 이유를 설명하지만, 상관관계(Correlation)와 샘플링 전략(Sampling Strategy)은 실제 잡음 수준을 변화시킬 수 있다.

작은 배치 학습(Small-Batch Training)은 더 강한 확률적 변동(Stochastic Fluctuation)을 도입하여 최적화 과정이 손실 지형의 여러 영역을 탐색하도록 도울 수 있다. 일부 조건에서는 이러한 변동이 유용한 일반화 특성(Generalization Property)을 가진 해를 선호하게 한다. 그러나 과도한 기울기 잡음은 수렴을 지연시키거나 학습을 불안정하게 만들고 유용한 기울기 방향(Informative Gradient Direction)을 가릴 수 있다. 따라서 작은 배치가 자동으로 더 우수한 모델을 생성하는 것은 아니다.

큰 배치 학습(Large-Batch Training)은 경험적 기울기(Empirical Gradient)를 더 정확하게 추정하며 병렬 GPU 연산(Parallel GPU Computation)을 효율적으로 활용할 수 있다. 이러한 특성은 대규모 신경망 학습(Large-Scale Neural Network Training)과 분산 시스템(Distributed System)에서 유용하다. 그러나 큰 배치는 서로 다른 최적화 경로를 따를 수 있으며, 학습률 스케줄과 학습 예산(Training Budget)을 적절히 조정하지 않으면 작은 배치에 비해 일반화 격차(Generalization Gap)가 나타날 수 있다.

배치 크기와 관련된 일반화 격차는 보편적인 법칙(Universal Law)이 아니다. 학습률 스케줄, 워밍업 절차(Warmup Procedure), 학습 지속 시간(Training Duration), 기타 하이퍼파라미터(Hyperparameter)를 신중하게 조정하면 큰 배치 모델도 경쟁력 있는 성능을 달성할 수 있다. 비교 과정에서는 배치 크기의 효과와 최적화 단계 수(Optimizer Step), 처리된 샘플 수(Processed Sample), 전체 계산량(Total Computation), 정규화 설정의 차이를 구분해야 한다.

배치 크기와 학습률은 확률적 갱신의 규모와 구조에 모두 영향을 주므로 밀접하게 상호작용한다. 배치 크기를 늘리면 기울기 잡음이 감소하는 반면, 학습률을 변경하면 매개변수 이동의 크기가 달라진다. 단순화된 가정(Simplified Assumption)에서는 근사적인 잡음 규모(Approximate Noise Scale)가 η/B와 관련되는 경우가 있다. 그러나 이러한 관계는 비선형 신경망 학습(Nonlinear Neural Network Training)에 적용되는 정확하고 보편적인 규칙은 아니다.

선형 학습률 스케일링(Linear Learning Rate Scaling)은 분산 학습(Distributed Training)에서 배치 크기를 증가시킬 때 자주 고려된다. 적절한 조건에서는 배치 크기를 특정 배수로 증가시키면 학습률도 비슷한 비율로 증가시키는 방법을 검토할 수 있다. 워밍업 스케줄(Warmup Schedule)은 초기 최적화 단계를 안정화하는 데 도움이 될 수 있다. 그러나 모델 구조(Model Architecture), 정규화 방식(Normalization), 최적화 알고리즘의 동작, 학습 체제(Training Regime)가 안정성에 영향을 주므로 스케일링 규칙은 실험적으로 검증해야 한다.

배치 정규화(Batch Normalization)는 배치 크기와 학습 동역학 사이에 추가적인 상호작용을 도입한다. 활성화 통계량(Activation Statistics)의 추정치는 각 배치에 포함된 샘플에 의존한다. 작은 배치는 잡음이 많은 통계량(Noisy Statistics)을 생성할 수 있지만, 큰 배치는 일반적으로 더 안정적인 추정치를 제공한다. 따라서 관측된 배치 크기 효과는 확률적 기울기 동작과 정규화 계층(Normalization Layer)의 동작 모두에서 발생할 수 있다.

기울기 누적(Gradient Accumulation)은 매개변수를 갱신하기 전에 여러 작은 마이크로배치(Micro-Batch)의 기울기를 결합하여 더 큰 유효 배치 크기(Effective Batch Size)를 모사한다. 손실값(Loss Value)의 크기를 올바르게 조정하면 모든 샘플을 동시에 GPU 메모리에 저장하지 않고도 큰 배치의 기울기를 근사할 수 있다. 그러나 배치에 의존하는 계층(Batch-Dependent Layer)과 확률적 연산(Stochastic Operation) 때문에 완전히 동일한 결과를 얻지 못할 수 있다.

파이토치(PyTorch)에서는 torch.optim.SGD를 사용하여 동일한 모델(Identical Model)을 학습하면서 학습률과 모멘텀을 변경함으로써 SGD 효과를 조사할 수 있다. 각 실험에서는 초기화 절차(Initialization Procedure), 데이터셋 분할(Dataset Split), 평가 지표(Evaluation Metric)를 동일하게 유지해야 한다. 학습 손실, 검증 정확도(Validation Accuracy), 기울기 노름(Gradient Norm), 수렴 속도(Convergence Speed)는 최적화 알고리즘 설정이 최적화와 일반화에 미치는 영향을 보여주는 근거가 된다.

배치 크기 실험(Batch-Size Experiment)은 torch.utils.data.DataLoader의 batch_size 인자를 변경하여 구현할 수 있다. 예를 들어 배치당 32개, 128개, 512개의 샘플을 사용하는 구성을 통제된 조건에서 비교할 수 있다. 배치 크기가 달라지면 에포크당 최적화 갱신 횟수(Optimizer Update Count)도 달라지므로 실험에서는 에포크 수(Epoch Count)와 최적화 단계 수(Optimization Step Count)를 모두 명시적으로 보고해야 한다.

엄격한 비교(Rigorous Comparison)에서는 동일 에포크 실험(Equal-Epoch Experiment), 동일 갱신 횟수 실험(Equal-Update Experiment), 동일 계산량 실험(Equal-Compute Experiment)을 구분해야 한다. 동일 에포크 학습은 거의 같은 수의 샘플을 처리하지만 큰 배치에서는 갱신 횟수가 적다. 동일 갱신 횟수 학습은 비슷한 수의 매개변수 갱신을 수행하지만 처리하는 데이터량은 다르다. 동일 계산량 비교는 계산 자원(Computational Resource)을 통제하여 학습 효율성(Training Efficiency)을 다른 관점에서 평가한다.

기울기 잡음은 동일한 모델 매개변수에서 독립적으로 샘플링한 여러 미니배치의 기울기를 측정하여 실험적으로 조사할 수 있다. 기울기의 변동성(Gradient Variability)은 확률적 기울기 분산(Stochastic Gradient Dispersion)을 추정하는 근거를 제공한다. 연구자는 기울기 공분산(Gradient Covariance), 헤시안 관련 곡률 지표(Hessian-Related Curvature Measure), 매개변수 민감도(Parameter Sensitivity)도 조사할 수 있다. 이러한 측정은 최적화 동작을 특성화하는 데 도움이 되지만 일반화를 직접 보장하는 지표로 해석해서는 안 된다.

암묵적 정규화는 가중치 감쇠(Weight Decay), 드롭아웃(Dropout), 데이터 증강(Data Augmentation), 조기 종료(Early Stopping) 같은 명시적 정규화 방법(Explicit Regularization Method)과 상호작용한다. 예를 들어 작은 배치는 확률적 기울기 잡음을 도입하는 반면, 가중치 감쇠는 매개변수 크기(Parameter Magnitude)를 직접 제한한다. 이러한 메커니즘은 서로 보완할 수 있지만 결합 효과가 반드시 가산적(Additive)인 것은 아니다. 최적화 알고리즘에서 발생하는 효과와 명시적 정규화 효과를 구분하려면 신중한 실험이 필요하다.

현대의 대규모 학습(Large-Scale Training)에서는 배치 크기가 GPU 메모리(GPU Memory), 통신 대역폭(Communication Bandwidth), 병렬 처리 효율성(Parallel Processing Efficiency)의 제약도 받는다. 분산 데이터 병렬 처리(Distributed Data Parallelism)는 여러 장치(Device)에 걸쳐 전역 배치 크기(Global Batch Size)를 증가시킬 수 있으며, 기울기 누적은 유효 배치 크기를 추가로 변경할 수 있다. 이러한 엔지니어링 선택(Engineering Choice)은 최적화 갱신 횟수에 영향을 주며 학습 과정의 암묵적 정규화 특성도 변화시킬 수 있다.

암묵적 정규화는 일반화가 모델 구조와 명시적인 벌점에만 의존하는 것이 아니라 매개변수가 최적화되는 방식에도 의존한다는 것을 보여준다. SGD 잡음(SGD Noise), 학습률 스케줄(Learning Rate Schedule), 모멘텀, 적응형 갱신(Adaptive Update), 배치 크기는 함께 작용하여 학습된 해를 형성한다. 이러한 상호작용을 이해하면 현대 기계학습 시스템(Modern Machine Learning System)에서 학습 안정성(Training Stability), 계산 효율성, 일반화에 관한 더욱 합리적인 선택을 할 수 있다.

## 03.07 Regularization in Modern AI

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

현대 인공지능(Modern Artificial Intelligence)에서 정규화(Regularization)는 모델 매개변수(Model Parameter)에 적용하는 전통적인 벌점(Penalty)을 넘어 다양한 방법으로 확장되고 있다. 현대 인공지능 시스템은 가중치 감쇠(Weight Decay), 드롭아웃(Dropout), 데이터 증강(Data Augmentation), 조기 종료(Early Stopping), 최적화 과정에서 발생하는 효과(Optimization-Induced Effect)를 결합하여 일반화(Generalization)를 향상한다. 이러한 방법은 높은 학습 성능(Training Performance)이 새로운 데이터에 대한 신뢰할 수 있는 동작을 보장하지 않는 대규모 신경망(Large Neural Network)에서 특히 중요하다.

현대적인 정규화 전략(Regularization Strategy)은 모델 구조(Model Architecture), 데이터셋 특성(Dataset Characteristic), 학습 목적(Training Objective) 사이의 관계를 고려해야 한다. 이미지 분류(Image Classification) 성능을 향상하는 기법이 언어 모델링(Language Modeling)이나 로봇 인식(Robotic Perception)에서 동일한 효과를 제공하지는 않을 수 있다. 따라서 효과적인 정규화를 위해서는 어떤 변화를 무시해야 하는지, 어떤 특징(Feature)을 구별 가능한 상태로 유지해야 하는지, 학습 절차(Training Procedure)가 학습된 표현(Learned Representation)에 어떤 영향을 주는지 이해해야 한다.

가중치 감쇠는 현대 딥러닝(Deep Learning)에서 가장 널리 사용되는 정규화 방법 중 하나이다. 과도한 매개변수 크기(Parameter Magnitude)를 억제함으로써 최적화(Optimization)에 영향을 주고 학습 데이터에만 존재하는 패턴(Training-Specific Pattern)에 대한 민감도를 줄일 수 있다. AdamW는 적응형 기울기 갱신(Adaptive Gradient Update)과 매개변수 축소(Parameter Shrinkage)를 분리하는 분리형 가중치 감쇠(Decoupled Weight Decay)를 구현한다. 기존 L2 벌점(L2 Penalty)은 적응형 최적화(Adaptive Optimization)에서 동일하게 동작하지 않으므로 이러한 차이는 중요하다.

실제 파이토치(PyTorch) 구현에서는 torch.optim.AdamW를 사용하여 선택한 학습률(Learning Rate)과 weight_decay 값을 설정할 수 있다. 서로 다른 매개변수 그룹(Parameter Group)에 다른 감쇠 설정(Decay Setting)을 적용할 수 있으며, 특히 편향(Bias)과 정규화 매개변수(Normalization Parameter)를 가중치 감쇠에서 제외하는 경우가 있다. 동일한 명목상 감쇠 계수(Nominal Decay Coefficient)라도 최적화 알고리즘(Optimizer)과 학습 구성(Training Configuration)에 따라 다른 효과를 나타낼 수 있으므로 이러한 선택을 문서화해야 한다.

드롭아웃은 특정 신경망 구조(Neural Network Architecture)에서 여전히 유용한 정규화 방법이다. 학습 중 일부 뉴런(Unit)을 무작위로 비활성화하여 특정 활성화 경로(Activation Pathway)에 대한 의존성을 줄이고 과도한 특징 공동 적응(Feature Co-Adaptation)을 억제한다. 그러나 모든 현대 신경망 구조에서 드롭아웃이 항상 유익한 것은 아니다. 효과는 네트워크 구조(Network Structure), 정규화 방법(Normalization Method), 학습 데이터량(Training Data Volume), 다른 정규화 기법과의 상호작용에 따라 달라진다.

트랜스포머(Transformer) 구조는 모델 규모(Model Scale)와 학습 방법론(Training Methodology)의 발전에 따라 정규화가 어떻게 변화했는지를 보여준다. 어텐션 드롭아웃(Attention Dropout), 잔차 드롭아웃(Residual Dropout), 가중치 감쇠, 확률적 깊이(Stochastic Depth)는 학습 동작(Training Behavior)에 영향을 줄 수 있다. 일부 대규모 트랜스포머 구성에서는 충분한 데이터와 다른 정규화 메커니즘(Regularization Mechanism)이 확보되면 드롭아웃을 거의 사용하지 않거나 생략한다. 따라서 정규화 설정은 다른 구조의 값을 그대로 복사하기보다 실험적으로 선택해야 한다.

대규모 언어 모델(Large Language Model, LLM)은 사전 학습 데이터셋(Pretraining Dataset)의 규모가 매우 크고 학습 목적이 토큰 분포(Token Distribution)를 예측하는 데 있으므로 추가적인 고려 사항이 필요하다. 가중치 감쇠, 데이터 품질 필터링(Data Quality Filtering), 학습률 스케줄링(Learning Rate Scheduling), 학습 지속 시간(Training Duration)은 일반화에 영향을 줄 수 있다. 일부 대규모 사전 학습 구성에서는 드롭아웃을 줄이거나 생략하지만, 작은 데이터셋으로 미세 조정(Fine-Tuning)할 때는 과적합(Overfitting)을 방지하기 위한 더 강한 제어가 필요할 수 있다.

사전 학습 모델(Pretrained Model)의 미세 조정은 처음부터 모델을 학습하는 경우와 다른 정규화 문제를 만든다. 사전 학습된 신경망은 이미 방대한 데이터에서 학습한 표현을 포함하고 있으므로 공격적인 매개변수 갱신(Aggressive Parameter Update)은 유용한 지식을 손상시킬 수 있다. 작은 학습률, 선택적 계층 동결(Selective Layer Freezing), 가중치 감쇠, 조기 종료는 사전 학습 능력(Pretrained Capability)을 보존하면서 새로운 도메인(Domain)이나 작업(Task)에 적응하도록 도울 수 있다.

저랭크 적응(Low-Rank Adaptation, LoRA)을 포함한 매개변수 효율적 미세 조정(Parameter-Efficient Fine-Tuning) 방법은 작은 적응 구성 요소(Compact Adaptation Component)를 도입하여 학습 가능한 매개변수(Trainable Parameter)의 수를 줄인다. 이러한 제한은 가능한 갱신 공간(Update Space)을 축소하여 경우에 따라 과적합을 줄일 수 있지만 보편적인 정규화 보장은 아니다. LoRA 구성에는 드롭아웃, 랭크 선택(Rank Selection), 스케일링 매개변수(Scaling Parameter)가 포함될 수 있으며, 이들은 적응 용량(Adaptation Capacity)에 영향을 준다.

데이터 증강은 실제 배포 환경(Deployment Environment)에 상당한 시각적 변동(Visual Variability)이 존재하므로 현대 컴퓨터 비전(Computer Vision)에서 특히 중요하다. 무작위 자르기(Random Cropping), 색상 변환(Color Transformation), 기하학적 교란(Geometric Perturbation), 믹스업(Mixup), 컷믹스(CutMix)는 학습 데이터의 다양성을 확장할 수 있다. 그러나 증강 정책(Augmentation Policy)은 레이블 유효성(Label Validity)과 작업별 제약(Task-Specific Constraint)을 유지해야 한다. 이미지 분류에 적합한 변환이 객체 감지(Object Detection), 분할(Segmentation), 기하학적 추정(Geometric Estimation)에는 부적절할 수 있다.

자기지도학습(Self-Supervised Learning)은 변환된 데이터 뷰(Transformed View)를 사용하여 학습 샘플 사이의 관계를 정의할 수 있으므로 데이터 증강을 다른 방식으로 활용한다. 대조 학습(Contrastive Learning)은 관련된 뷰의 표현이 서로 유사하게 유지되도록 하면서 관련 없는 샘플을 구별하도록 유도하는 경우가 많다. 변환의 선택은 어떤 속성이 불변성(Invariance)을 갖게 되는지 결정한다. 지나치게 강한 데이터 증강은 후속 응용(Downstream Application)에서 보존해야 할 정보를 제거할 수 있다.

현대 객체 감지 시스템(Modern Object Detection System)은 분류(Classification)와 위치 추정(Localization) 목적 함수를 결합하므로 정규화 기법을 신중하게 조정해야 한다. 이미지 변환(Image Transformation)은 경계 상자(Bounding Box)를 일관되게 갱신해야 하며, 강한 데이터 증강은 객체 가시성(Object Visibility)과 크기(Scale)를 변화시킬 수 있다. 가중치 감쇠, 조기 종료, 적절한 증강 정책은 강건성(Robustness)을 개선할 수 있다. 효과는 분류 정확도(Classification Accuracy)만이 아니라 객체 감지 지표(Detection Metric)를 사용하여 측정해야 한다.

의미론적 분할(Semantic Segmentation)은 모델이 상세한 공간 정보(Spatial Information)를 보존해야 하므로 추가적인 요구사항이 있다. 무작위 자르기, 광도 변환(Photometric Transformation), 크기 증강(Scale Augmentation)은 일반화를 개선할 수 있지만, 기하학적 변환(Geometric Transformation)은 이미지와 분할 마스크(Segmentation Mask) 사이의 정렬(Alignment)을 유지해야 한다. 지나친 평활화(Smoothing)나 공간 왜곡(Spatial Distortion)은 특히 작은 객체(Small Object)나 희귀 클래스(Rare Class)가 중요한 경우 경계 정확도(Boundary Accuracy)를 저하시킬 수 있다.

로봇 인식 시스템(Robotic Perception System)에는 실제 센서 특성(Real Sensor Characteristic)과 운영 조건(Operating Condition)을 반영하는 정규화 전략이 필요하다. 카메라 잡음(Camera Noise), 조명 변화(Illumination Variation), 부분 가림(Partial Occlusion), 환경 변화(Environmental Change)는 신중하게 설계된 데이터 증강으로 표현할 수 있다. 그러나 임의의 변환은 물리적 제약(Physical Constraint)이나 센서 보정(Sensor Calibration)을 위반할 수 있다. 따라서 학습 강건성(Training Robustness)은 현실적인 배포 시나리오(Deployment Scenario)를 기준으로 평가해야 한다.

다중모달 모델(Multimodal Model)에서는 정규화가 서로 다른 입력 모달리티(Input Modality) 사이의 관계를 보존해야 한다. 이미지, 언어, 오디오, 센서 표현(Sensor Representation)을 공동으로 학습할 수 있으므로 일관되지 않은 데이터 증강(Inconsistent Augmentation)은 해로울 수 있다. 예를 들어 이미지와 관련된 공간 주석(Spatial Annotation)을 갱신하지 않고 이미지만 수정하면 모순된 감독 신호(Contradictory Supervision)가 발생할 수 있다. 효과적인 다중모달 정규화(Multimodal Regularization)는 개별 모달리티와 모달리티 간 일관성(Cross-Modal Consistency)을 모두 고려한다.

확산 모델(Diffusion Model)과 기타 생성형 구조(Generative Architecture)는 단순히 판별 경계(Discriminative Boundary)를 학습하는 것이 아니라 복잡한 데이터 분포(Data Distribution)를 학습하므로 추가적인 정규화 문제가 존재한다. 학습 목적, 잡음 스케줄(Noise Schedule), 데이터셋 다양성(Dataset Diversity), 가중치 감쇠는 생성 결과(Generated Output)에 영향을 줄 수 있다. 지나친 정규화는 충실도(Fidelity)나 표현 용량(Representation Capacity)을 감소시킬 수 있으며, 제어가 부족하면 암기(Memorization)와 불안정한 학습(Unstable Training)이 발생할 수 있다.

조기 종료는 현대 인공지능에서도 유용하며, 특히 미세 조정과 제한된 데이터셋을 사용하는 실험에서 중요하다. 검증 손실(Validation Loss)이나 작업별 지표(Task-Specific Metric)를 사용하면 마지막 학습 반복(Training Iteration)보다 더 우수한 성능을 나타내는 체크포인트(Checkpoint)를 찾을 수 있다. 그러나 대규모 사전 학습은 계획된 토큰 예산(Token Budget)이나 계산 예산(Compute Budget)에 따라 진행되는 경우가 많다. 이러한 환경에서는 전통적인 인내 횟수 기반 종료(Patience-Based Termination)보다 체크포인트 선택(Checkpoint Selection)과 학습 스케줄(Training Schedule)이 더 실용적일 수 있다.

레이블 스무딩(Label Smoothing)은 현대 인공지능 시스템의 분류 구성 요소(Classification Component)에 사용되는 또 다른 정규화 방법이다. 정확한 원-핫 목표값(One-Hot Target)을 완화된 확률 분포(Softened Probability Distribution)로 대체하여 극단적인 확신도(Extreme Confidence)를 향한 압력을 줄인다. 그러나 모든 모델 구조나 목적 함수(Objective Function)에 평활화가 유익한 것은 아니다. 정확도(Accuracy), 확률 보정(Calibration), 표현 학습(Representation Learning)에 미치는 영향은 보편적으로 긍정적이라고 가정하기보다 독립적으로 평가해야 한다.

암묵적 정규화(Implicit Regularization)는 최적화 알고리즘 선택(Optimizer Selection), 배치 크기(Batch Size), 학습률 스케줄, 초기화(Initialization)를 통해 계속 중요한 역할을 수행한다. 확률적 경사하강법(Stochastic Gradient Descent, SGD)과 적응형 최적화 알고리즘(Adaptive Optimizer)은 유사한 목적 함수로 학습하더라도 서로 다른 최적화 경로(Optimization Trajectory)를 선호할 수 있다. 큰 배치 학습(Large-Batch Training)에는 학습률 조정(Learning Rate Adjustment)과 워밍업 스케줄(Warmup Schedule)이 필요할 수 있다. 이러한 상호작용은 정규화가 전체 학습 절차에 의해 부분적으로 결정된다는 사실을 보여준다.

실용적인 이미지 분류 실험(Practical Image Classification Experiment)에서는 기준 합성곱 신경망(Baseline Convolutional Neural Network, CNN)과 가중치 감쇠, 드롭아웃, 데이터 증강을 적용한 구성을 비교할 수 있다. 실험 전반에 걸쳐 동일한 학습 및 검증 데이터 분할(Training and Validation Split)을 유지해야 한다. 정확도, 검증 손실, 학습 시간(Training Duration), 일반화 격차(Generalization Gap)를 기록하면 통제된 조건에서 어떤 기법이 측정 가능한 성능 개선에 기여하는지 확인할 수 있다.

트랜스포머 미세 조정(Transformer Fine-Tuning)에서는 전체 매개변수 갱신(Full Parameter Update)과 선택적 계층 동결 및 매개변수 효율적 적응(Parameter-Efficient Adaptation)을 비교하는 실험을 수행할 수 있다. 각 구성에는 비교 가능한 검증 절차(Validation Procedure)를 적용하고 학습 가능한 매개변수 수(Trainable Parameter Count)를 보고해야 한다. 메모리 사용량(Memory Consumption), 수렴 속도(Convergence Speed), 후속 작업 성능(Downstream Performance)을 추가로 측정하면 효율성 향상이 허용 가능한 일반화 성능을 동반하는지 판단할 수 있다.

정규화 실험(Regularization Experiment)에서는 여러 변수를 동시에 변경하지 않아야 한다. 변수 사이의 상호작용이 개별 방법의 기여도를 불분명하게 만들 수 있기 때문이다. 제거 실험(Ablation Study)은 다른 설정을 통제하면서 선택한 구성 요소를 체계적으로 제거하거나 수정한다. 예를 들어 데이터 증강만 적용한 경우, 가중치 감쇠만 적용한 경우, 두 기법을 결합한 경우를 비교하면 효과가 상호 보완적(Complementary)인지 중복적(Redundant)인지 확인할 수 있다.

정규화 강도(Regularization Strength)는 데이터셋과 학습 체제(Training Regime)에 따라 달라지므로 하이퍼파라미터 선택(Hyperparameter Selection)이 필수적이다. 가중치 감쇠 계수(Weight Decay Coefficient), 드롭아웃 확률(Dropout Probability), 데이터 증강 강도(Augmentation Intensity), 조기 종료 인내 횟수(Early Stopping Patience)는 모두 모델 동작에 영향을 준다. 과도한 정규화는 과소적합(Underfitting)을 일으킬 수 있으며, 부족한 정규화는 암기를 허용할 수 있다. 검증 실험(Validation Experiment)에서는 예측 성능(Predictive Performance)과 학습 안정성(Training Stability)을 함께 조사해야 한다.

모델이 복잡한 환경에서 작동한다면 평가는 단일 정확도 지표(Single Accuracy Metric)를 넘어 확장되어야 한다. 확률 보정, 분포 변화에 대한 강건성(Robustness to Distribution Shift), 희귀 클래스 성능(Rare-Class Performance), 계산 효율성(Computational Efficiency)도 동일하게 중요할 수 있다. 안전 민감 응용(Safety-Sensitive Application)에서는 현실적인 교란(Realistic Disturbance)과 고장 조건(Failure Condition)을 검증에 포함해야 한다. 정규화는 그 이점이 목표 시스템(Intended System)의 성능 요구사항(Performance Requirement)에 실제로 반영될 때 가치가 있다.

현대 정규화 방법을 비교할 때 재현성(Reproducibility)은 특히 중요하다. 실험에서는 난수 시드(Random Seed), 데이터셋 버전(Dataset Version), 최적화 알고리즘 설정(Optimizer Setting), 학습률 스케줄, 배치 크기, 데이터 증강 정책을 기록해야 한다. 여러 차례의 학습 실행(Training Run)은 일관된 성능 개선과 무작위 변동(Random Variation)을 구분하는 데 도움이 된다. 오해를 유발하는 비교를 방지하기 위해 체크포인트 선택 절차(Checkpoint Selection Procedure)도 문서화해야 한다.

현대 정규화는 독립적인 기법의 집합이라기보다 조정된 학습 설계(Coordinated Training Design)로 이해하는 것이 가장 적절하다. 명시적 벌점(Explicit Penalty), 확률적 최적화(Stochastic Optimization), 데이터 변환(Data Transformation), 구조적 제약(Architectural Constraint), 검증 절차는 함께 작용하여 학습된 해(Learned Solution)에 영향을 준다. 효과적인 구현을 위해서는 신뢰할 수 있는 실제 환경 예측(Real-World Prediction)에 필요한 정보를 보존하면서 모델 용량(Model Capacity), 계산 비용(Computational Cost), 일반화 사이의 균형을 유지해야 한다.
