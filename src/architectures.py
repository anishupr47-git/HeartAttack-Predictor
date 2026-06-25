import numpy as np
#train test split is important here
#we train test and validate here
from sklearn.model_selection import train_test_split

#these are metric functions
#we check accuracy
#we check f1 score
#and check in confusion matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score

#machine learning algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

#deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def build_dnn_module(input_dim):
    """
    Build a Deep Neural Network
    """
    #we use sequential model
    model = Sequential([
        #Layer 1 256 input dims
        Dense(256, input_dim=input_dim),
        BatchNormalization(), #keeps numbers balanced
        tf.keras.layers.Activation('relu'), #pass only +ve nums
        Dropout(0.3), #reduces overfitting
        
        #Layer 2
        Dense(128),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),  # Added missing comma here
        Dropout(0.3),

        #Layer 4
        Dense(64),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),  # Added missing comma here

        #Layer 5
        Dense(32),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),

        #final output
        Dense(1, activation='sigmoid') #sigmoid turns final num in probability 0 and 1
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_all_models():
    """
    Returns a dict to every ML model
    """
    return {
        #XGBOOST builds many decision trees in sequence learning from previous error
        'XGBoost': xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, eval_metric='logloss', random_state=42),

        #Randomforest builds many decisionm trees independtly for one final answer
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', max_features='sqrt', random_state=42),

        #Logistic regression draws a straight line to seperate classes
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),

        #Decision trees act a set of yes or no qs
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),

        #SVC finds the optimal boundary between class in high dimensions
        'SVM': SVC(probability=True, random_state=42),

        #KNN looks at the closest 5 data points to take a major vote
        'KNN': KNeighborsClassifier(n_neighbors=5),

        #Gaussian naive bayes use probability theory
        'Gaussian Naive Bayes': GaussianNB(),

        #ADABOOST creates a strong model by combining the weak ones
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),

        #Extratress is like random forest but choose split points
        'Extra Trees': ExtraTreesClassifier(n_estimators=300, random_state=42),

        #Gradient Boosting is another sequential tree builder
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    }

def build_stacking_ensemble(models_dict):
    """
    Builds a voting classifier
    """
    estimators = [
        ('rf', models_dict['Random Forest']),
        ('xgb', models_dict['XGBoost']),
        ('lr', models_dict['Logistic Regression'])
    ]
    return VotingClassifier(estimators=estimators, voting='soft')

def master_training_orchestrator(X, y):
    """
    The master training loop
    """
    #split the data for train
    #80% and 20%
    X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    metrics = {}
    trained_models= {}

    #training in dnn
    print("Training 1/12: Deep Neural Network")
    dnn_model = build_dnn_module(input_dim=X_train.shape[1])
    
    #early stop if the models aint better
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #.fit() to start the learning process
    dnn_model.fit(X_train, y_train,validation_data=(X_val,y_val), epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    #.predict() ask to model to guess
    dnn_preds_prob = dnn_model.predict(X_val).flatten()
    dnn_preds = (dnn_preds_prob > 0.5).astype(int)

    #we save all score
    metrics['DNN'] = {
        'Accuracy': float(accuracy_score(y_val, dnn_preds)),
        'F1-Score': float(f1_score(y_val, dnn_preds)),
        'Precision': float(precision_score(y_val, dnn_preds)),
        'Recall': float(recall_score(y_val, dnn_preds)),
        'ROC-AUC': float(roc_auc_score(y_val, dnn_preds_prob)),
        'Confusion_Matrix': confusion_matrix(y_val, dnn_preds).tolist()
    }
    trained_models['DNN'] = dnn_model

    #Training models 2 to 11
    ml_models= get_all_models()

    #we use a loop to go to all
    for i, (name,model) in enumerate(ml_models.items(), start=2):
        print(f"Training {i}/12: {name}")

        #train the model
        model.fit(X_train, y_train)

        #make prediction
        preds= model.predict(X_val)

        #calculate probability
        if hasattr(model, 'predict_proba'):
            probs= model.predict_proba(X_val)[:,1]
            roc_auc = float(roc_auc_score(y_val, probs))
        else:
            roc_auc = float(roc_auc_score(y_val, preds))

        #save all models
        metrics[name] = {
            'Accuracy': float(accuracy_score(y_val, preds)),
            'F1-Score': float(f1_score(y_val, preds)),
            'Precision': float(precision_score(y_val, preds, zero_division=0)),
            'Recall': float(recall_score(y_val, preds, zero_division=0)),
            'ROC-AUC': roc_auc,
            'Confusion_Matrix': confusion_matrix(y_val, preds).tolist()
        }
        trained_models[name] = model

    #Training voting
    print("Training 12/12: Soft voting ensemble")
    ensemble = build_stacking_ensemble(ml_models)
    ensemble.fit(X_train,y_train)

    ens_preds = ensemble.predict(X_val)
    ens_probs = ensemble.predict_proba(X_val)[:,1]

    metrics['Voting Ensemble'] = {
        'Accuracy': float(accuracy_score(y_val, ens_preds)),
        'F1-Score': float(f1_score(y_val, ens_preds)),
        'Precision': float(precision_score(y_val, ens_preds)),
        'Recall': float(recall_score(y_val, ens_preds)),
        'ROC-AUC': float(roc_auc_score(y_val, ens_probs)),
        'Confusion_Matrix': confusion_matrix(y_val, ens_preds).tolist()
    }
    trained_models['Voting Ensemble'] = ensemble

    print("All 12 trained and evaluated successfully")

    #we return architecture
    return trained_models, metrics, X_train, X_val
