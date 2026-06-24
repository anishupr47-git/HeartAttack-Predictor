import shap
import numpy as np
import plotly.graph_objects as go

#shap explanability
#shap opens the black box as ml is known as black box too
#it uses game theory to figure out exactly how much each feature is contributed about final prediction

def generate_shap_explanation(model, X_train, input_scaled, feature_names, model_name):
    """
    Generate shap value for specific patient prediction
    """
    print("Selecting 100 background samples for SHAP baseline")
    background = shap.sample(X_train, 100)

    #different algorithm need diff tools
    #tree algorithms like random forest
    #other model use kernerexplainers

    print(f"Figuring out which explainer to use for {model_name}")

    if model_name in ['XGBoost', 'Random Forest', 'Extra Trees', 'Gradient Boosting']:
        print("This is a Tress-based model using treeExplainer")
        try:
            #Treeexpaliner is very fast and for decision tree
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)

            #sometimes shap returns a list of values
            #we always want index [1]
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv = shap_values[0]
                expected_value = explainer.expected_value
        except Exception:
            #if treeExplainer fails, we safely fall back to kernerexplainer
            print("TreeExplainer failed, failing back to KernelExplainer")
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(input_scaled)
            sv = shap_values[0]
            expected_value = explainer.expected_value
    
    elif model_name == 'DNN':
        print("This is a Deep Neural Network Using KernelExplainer")
        #deep xplainer can be very buggy on tf verisons
        #so swe use kernelexplainer
        explainer= shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(input_scaled)

        #neural networks reutrn list of lists, we just need the first number
        sv= shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]
        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    else:
        print("This is a Standard Model for using logistic regression etc using kernelexplainer")
        #for standard models, we explain probability output
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(input_scaled)

        #extract the values for predicting class 1
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            expected_value = explainer.expected_value[1]
        else:
            sv = shap_values[0][:, 1] if shap_values[0].ndim >1 else shap_values[0]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value