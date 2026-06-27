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

            #unified shap parsing logic
            #parse to shap value for contributions
        if isinstance(shap_values, list):
            #we want for class 1 or main for 0
            if len(shap_values) > 1:
                sv = shap_values[1]
            else:
                sv=shap_values[0]
            #if the array is 2d extract the first sample
            if isinstance(sv, np.ndarray) and sv.ndim == 2:
                sv = sv[0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim ==3:
                    if shap_values.shape[2]>1:
                        sv=shap_values[0, :, 1]
                    else:
                        sv=shap_values[0,:,0]
                elif shap_values.ndim==2:
                    sv=shap_values[0]
                else:
                    sv = shap_values
        else:
            sv=shap_values

        #parse to get 1 
        if isinstance(expected_value, (list, np.ndarray)):
            #if has more than two
            if len(expected_value) > 1:
                expected_value = expected_value[1]
            elif len(expected_value) > 0:
                expected_value = expected_value[0]


    #Building waterfall chart
    print("Shap values calculated, building waterfall chart")
    
    #make sure to make it a flat list of nums
    sv = np.array(sv).flatten()
    expected_value = float(expected_value)

    #we show the top 10 values to make it valid
    #np.argsort sorts number from low to high
    #np.abs() check absolute value
    #[-10:] gives the last 10 values
    indices = np.argsort(np.abs(sv))[-10:]

    #grab the info of the last 10
    top_features = [feature_names[i] for i in indices]
    top_sv = sv[indices]

    other_sv = np.sum(sv) - np.sum(top_sv)

    chart_x = ["Base Value"] + top_features + ["Other Features", "Prediction"]
    chart_y = [expected_value] + list(top_sv) + [other_sv, 0]
    measure = ["absolute"] + (["relative"] * len(top_sv)) + ["relative", "total"]

    final_pred = expected_value + np.sum(sv)
    #prepare the text for chat
    text = [f"{expected_value:.3f}"] + [f"{val:+.3f}" for val in top_sv] + [f"{other_sv:+.3f}", f"{final_pred:.3f}"]

    #build plotly figure
    fig = go.Figure(go.Waterfall(
        name="SHAP Explainability",
        orientation="v",
        measure=measure,
        x=chart_x,
        textposition="outside",
        text=text,
        y=chart_y,
        connector={"line": {"color": "rgb(63, 63, 63)"}}, 
        decreasing={"marker": {"color": "MediumTurquoise"}}, 
        increasing={"marker": {"color": "Crimson"}},         
        totals={"marker": {"color": "blue"}}
    ))

    #Final Touches to the chart
    fig.update_layout(
        title=f"Prediction Explainability (SHAP Top 10 features) - Base Value: {float(expected_value):.3f}",
        showlegend=False,
        height=500,
        template='plotly_white',
        yaxis_title="Contribution to Prediction Risk"
    )

    print("Waterfall chart ready")
    return fig