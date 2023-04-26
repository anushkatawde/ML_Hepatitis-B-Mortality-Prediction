# ---pip install pipreqs---
# to download requirement file
# pipreqs ./
# Core Pkgs
import pickle
from pathlib import Path
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import streamlit_authenticator as stauth
from PIL import Image

# ML Interpretation
import lime
import lime.lime_tabular

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Hepatitis Mortality App",page_icon="⚕️",layout="wide")
html_temp = """
		<div style="background-color:darkblue;padding:10px;border-radius:5px">
		<h1 style="color:white;text-align:center;"> Disease Mortality Prediction </h1>
		<h3 style="color:white;text-align:center;"> "Hepatitis B" </h3>
		</div>
		"""

result_temp2 = """
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probabilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp = """
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""

descriptive_message_temp = """
	<div style="background-color:lightgray;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<p style="text-align:justify">Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease i.e. Hepatitis B can range from a mild illness, lasting a few weeks, to a serious, 
		life-long (chronic) condition.
	    Hepatitis B is primarily spread when blood, semen, or certain other body fluids – even in microscopic amounts – from a person infected with the hepatitis B virus enters the body of someone who is not infected.
		Acute hepatitis B is a short-term illness that occurs within the first 6 months after someone is exposed to the hepatitis B virus. Some people with acute hepatitis B have no symptoms at all or only mild illness.
		For others, acute hepatitis B causes a more severe illness that requires hospitalization.<br>
		<br>
		The hepatitis B virus can also be transmitted by :-</p>
		<ul>
		<li style="text-align:justify;color:black">Birth to an infected pregnant person.</li>
		<li style="text-align:justify;color:black">Sex with an infected person.</li>
		<li style="text-align:justify;color:black">Sharing personal items such as toothbrushes or razors, but is less common.</li>
		<li style="text-align:justify;color:black">Poor infection control in health care facilities.</li>
		<li style="text-align:justify;color:black">Direct contact with the blood or open sores of a person who has hepatitis B, etc.</li>
		<ul>
        </div>
	"""




names = ["Sakshi Yadav", "Anushka Tawde"]
usernames = ["sYadav", "aTawde"]

feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites', 'varices', 'bilirubin',
                      'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']

gender_dict = {"male": 1, "female": 2}
feature_dict = {"No": 1, "Yes": 2}
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """Hep Mortality Prediction App"""

    st.sidebar.title("Hepatitis B Disease App")
    st.markdown(html_temp.format('royalblue'), unsafe_allow_html=True)

    menu = ["Home", "Login"]
    sub_menu = ["Plot", "Prediction"]
    st.sidebar.header(f"Please Filter Here:")
    choice = st.sidebar.selectbox("Select pages:", menu)
    if choice == "Home":
        st.write("")
        st.write("")
        #Loading image
        image = Image.open('hepDis.jpg')
        st.image(image, caption='Human Liver')
        st.write("")
        st.markdown("<h2 >What is Hepatitis?</h2>", unsafe_allow_html=True)
        st.markdown(descriptive_message_temp, unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")

        image2 = Image.open('hepimage.jpg')
        st.image(image2, caption='Symptoms of hepatitis')
        #st.image(load_image('hepimage.jpg'))


    elif choice == "Login":
        #username = st.sidebar.text_input("Username")
        #password = st.sidebar.text_input("Password", type='password')
        #if st.checkbox("Login"):
            file_path = Path(__file__).parent / "hashed_pws.pkl"
            with file_path.open("rb") as file:
                hashed_passwords = pickle.load(file)

            authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                                "sales_dashboard", "abcdef",cookie_expiry_days=30)

            name, authentication_status, username = authenticator.login("Login", "main")

            if authentication_status == False:
                st.error("Username/password is incorrect")

            if authentication_status == None:
                st.warning("Please enter your username and password")

            if authentication_status:
                authenticator.logout("Logout", "sidebar")
                st.sidebar.title(f"Welcome {name}")

                activity = st.sidebar.selectbox("Activity", sub_menu)
                if activity == "Plot":

                    st.markdown("<h2 style='text-align: center; color: darkblue;'>Data Visualization </h2>",
                                unsafe_allow_html=True)

                    df = pd.read_csv("data/clean_hepatitis_dataset.csv")
                    if st.checkbox("View Dataset"):
                        st.dataframe(df)
                    st.write("#### Below is the description of our Dataset:")
                    st.write(df.describe())

                    st.markdown("""---""")

                    st.write("#### Bar Plot: ")
                    st.write("feature 'class' = {'Die':1, 'Live':2} ")
                    df['class'].value_counts().plot(kind='bar')
                    st.pyplot()

                    st.markdown("""---""")

                    # Freq Dist Plot
                    freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")

                    st.write("#### Frequency Distribution Plot")
                    st.write("based on 'age' variable")
                    if st.checkbox("View groupwise Count of the above:"):
                        st.dataframe(freq_df)

                    st.bar_chart(freq_df['count'])
                    st.markdown("""---""")
                    if st.checkbox("Area Chart"):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect("Choose a Feature", all_columns)
                        new_df = df[feat_choices]
                        st.write(f"##### Area chart ")
                        st.area_chart(new_df)

                elif activity == "Prediction":
                    st.subheader("Predictive Analytics")

                    age = st.number_input("Age", 7, 80)
                    sex = st.radio("Sex", tuple(gender_dict.keys()))
                    steroid = st.radio("Do You Take Steroids?", tuple(feature_dict.keys()))
                    antivirals = st.radio("Do You Take Antivirals?", tuple(feature_dict.keys()))
                    fatigue = st.radio("Do You Have Fatigue", tuple(feature_dict.keys()))
                    spiders = st.radio("Presence of Spider Naeve", tuple(feature_dict.keys()))
                    ascites = st.selectbox("Ascities", tuple(feature_dict.keys()))
                    varices = st.selectbox("Presence of Varices", tuple(feature_dict.keys()))
                    bilirubin = st.number_input("bilirubin Content", 0.0, 8.0)
                    alk_phosphate = st.number_input("Alkaline Phosphate Content", 0.0, 296.0)
                    sgot = st.number_input("Sgot", 0.0, 648.0)
                    albumin = st.number_input("Albumin", 0.0, 6.4)
                    protime = st.number_input("Prothrombin Time", 0.0, 100.0)
                    histology = st.selectbox("Histology", tuple(feature_dict.keys()))
                    feature_list = [age, get_value(sex, gender_dict), get_fvalue(steroid), get_fvalue(antivirals),
                                    get_fvalue(fatigue), get_fvalue(spiders), get_fvalue(ascites), get_fvalue(varices),
                                    bilirubin, alk_phosphate, sgot, albumin, int(protime), get_fvalue(histology)]
                    st.write(len(feature_list))
                    st.write(feature_list)
                    pretty_result = {"age": age, "sex": sex, "steroid": steroid, "antivirals": antivirals,
                                     "fatigue": fatigue, "spiders": spiders, "ascites": ascites, "varices": varices,
                                     "bilirubin": bilirubin, "alk_phosphate": alk_phosphate, "sgot": sgot,
                                     "albumin": albumin, "protime": protime, "histolog": histology}
                    st.json(pretty_result)
                    single_sample = np.array(feature_list).reshape(1, -1)

                    # ML
                    model_choice = st.selectbox("Select Model", ["LR", "KNN", "DecisionTree"])
                    if st.button("Predict"):
                        if model_choice == "KNN":
                            loaded_model = load_model("models/knn_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        elif model_choice == "DecisionTree":
                            loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        else:
                            loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)

                        # st.write(prediction)
                        # prediction_label = {"Die":1,"Live":2}
                        # final_result = get_key(prediction,prediction_label)
                        if prediction == 1:
                            st.warning("Patient Dies")
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)
                            st.subheader("Prescriptive Analytics")
                            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Patient Lives")
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)

                    if st.checkbox("Interpret"):
                        if model_choice == "KNN":
                            loaded_model = load_model("models/knn_hepB_model.pkl")

                        elif model_choice == "DecisionTree":
                            loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")

                        else:
                            loaded_model = load_model("models/logistic_regression_hepB_model.pkl")

                            # loaded_model = load_model("models/logistic_regression_model.pkl")
                            # 1 Die and 2 Live
                            df = pd.read_csv("data/clean_hepatitis_dataset.csv")
                            x = df[['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites', 'varices',
                                    'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']]
                            feature_names = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites',
                                             'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime',
                                             'histology']
                            class_names = ['Die(1)', 'Live(2)']
                            explainer = lime.lime_tabular.LimeTabularExplainer(x.values, feature_names=feature_names,
                                                                               class_names=class_names,
                                                                               discretize_continuous=True)
                            # The Explainer Instance
                            exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,
                                                             num_features=13, top_labels=1)
                            exp.show_in_notebook(show_table=True, show_all=False)
                            # exp.save_to_file('lime_oi.html')
                            st.write(exp.as_list())
                            new_exp = exp.as_list()
                            label_limits = [i[0] for i in new_exp]
                            # st.write(label_limits)
                            label_scores = [i[1] for i in new_exp]
                            plt.barh(label_limits, label_scores)
                            st.pyplot()
                            plt.figure(figsize=(20, 10))
                            fig = exp.as_pyplot_figure()
                            st.pyplot()


if __name__ == '__main__':
    main()
