import pickle as pk
import requests
import streamlit as st
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
from streamlit_chat import message as st_message
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration





#loading diabetics model

#css
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return r.json()

hide_st_style = """
                <style>
                #MainMenu { visibility : hidden;}
                footer { visibility : hidden;}
                header { visibility : hidden;}
                </style>
                """
st.markdown(hide_st_style , unsafe_allow_html = True)
                

# use local css
def local_css(file_name):
    with open(file_name) as f :
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")



#images
img_home = Image.open("images/Diabetes.jpg")
img_info1 = Image.open("images/info1.jpg")
img_info2 = Image.open("images/info2.jpg")
img_info3 = Image.open("images/info3.webp")

diabetic_model = pk.load(open('diabetes_model.sav','rb'))

#with menu bar
selected = option_menu(
        menu_title ="",
        options =["home","Diabetes?","Diaognysis","Help_Bot"],
        icons = ["house","globe","globe","book"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal")

if selected == "Diaognysis":
    with st.container():
        def add_bg_from_url():
            st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://akm-img-a-in.tosshub.com/indiatoday/images/story/202206/diabetes.jpg?VersionId=m8MNZaJMHPgAraLrCSxe_hhuC5fFZxYX?size=1200:675");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
            unsafe_allow_html=True)
        add_bg_from_url()
    st.title("Diabetic prediction using Machine learning")
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetic_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)

if selected == "home":
    with st.container():
        def add_bg_from_url():
            st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.endocrine.org/-/media/endocrine/images/patient-engagement-webpage/condition-page-images/diabetes-and-glucose-metabolism/diabetes_treatments_pe_1796x9432.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
            unsafe_allow_html=True)
        add_bg_from_url()

if selected =="Help_Bot":
    with st.container():
        st.title("Talk with RK")
        def add_bg_from_url():
            st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://d2vrvpw63099lz.cloudfront.net/pharmacy-chatbot/header1.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
            unsafe_allow_html=True)
        add_bg_from_url()
    @st.experimental_singleton
    def get_models():
        
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
          model_name = "facebook/blenderbot-400M-distill"
          tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
          model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
          return tokenizer, model


    if "history" not in st.session_state:
           st.session_state.history = []

           st.title("Hello Chatbot")


    def generate_answer():
           tokenizer, model = get_models()
           user_message = st.session_state.input_text
           inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
           result = model.generate(**inputs)
           message_bot = tokenizer.decode(
           result[0], skip_special_tokens=True)
           # .replace("<s>", "").replace("</s>", "")

           st.session_state.history.append({"message": user_message, "is_user": True})
           st.session_state.history.append({"message": message_bot, "is_user": False})
    st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)
    for i, chat in enumerate(st.session_state.history):
                 st_message(**chat, key=str(i)) #unpacking


if selected == "Diabetes?":
    st.title("DIABETES")
    st.subheader("What is Diabetes ?")
    st.image(img_info1)
    st.write(
    """
Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.
Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream.
When your blood sugar goes up, it signals your pancreas to release insulin.
Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.
With diabetes, your body doesn’t make enough insulin or can’t use it as well as it should.
When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream.
Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.
There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help.
""")
    st.write("""
The number of people with diabetes rose from 108 million in 1980 to 422 million in 2014.
Prevalence has been rising more rapidly in low- and middle-income countries than in high-income countries.
Diabetes is a major cause of blindness, kidney failure, heart attacks, stroke and lower limb amputation.
Between 2000 and 2019, there was a 3% increase in diabetes mortality rates by age.
In 2019, diabetes and kidney disease due to diabetes caused an estimated 2 million deaths.
A healthy diet, regular physical activity, maintaining a normal body weight and avoiding tobacco use are ways to prevent or delay the onset of type 2 diabetes.
Diabetes can be treated and its consequences avoided or delayed with diet, physical activity, medication and regular screening and treatment for complications.
""")
    st.write("----")
    st.subheader("Types of Diabetes :")
    st.image(img_info2)
    st.subheader("Type 1")
    st.write("""
Type 1 diabetes (previously known as insulin-dependent, juvenile or childhood-onset) is characterized by deficient insulin production and requires daily administration of insulin.
In 2017 there were 9 million people with type 1 diabetes; the majority of them live in high-income countries.
Neither its cause nor the means to prevent it are known.
""")
    st.subheader("Type 2")
    st.write("""
Type 2 diabetes affects how your body uses sugar (glucose) for energy.
It stops the body from using insulin properly, which can lead to high levels of blood sugar if not treated.
Over time, type 2 diabetes can cause serious damage to the body, especially nerves and blood vessels.
Type 2 diabetes is often preventable.
Factors that contribute to developing type 2 diabetes include being overweight, not getting enough exercise, and genetics.
Early diagnosis is important to prevent the worst effects of type 2 diabetes.
The best way to detect diabetes early is to get regular check-ups and blood tests with a healthcare provider.
Symptoms of type 2 diabetes can be mild. They may take several years to be noticed.
Symptoms may be similar to those of type 1 diabetes but are often less marked.
As a result, the disease may be diagnosed several years after onset, after complications have already arisen.
""")
    st.write("---")
    st.subheader("Treatment")
    st.image(img_info3)
    st.write("""
Treatment for type 1 diabetes :

involves insulin injections or the use of an insulin pump, frequent blood sugar checks, and carbohydrate counting.
For some people with type 1 diabetes, pancreas transplant or islet cell transplant may be an option.

Treatment of type 2 diabetes :

mostly involves lifestyle changes, monitoring of your blood sugar, along with oral diabetes drugs, insulin or both.

Depending on what type of diabetes you have, blood sugar monitoring, insulin and oral drugs may be part of your treatment.
Eating a healthy diet, staying at a healthy weight and getting regular physical activity also are important parts of managing diabetes.
An important part of managing diabetes — as well as your overall health — is keeping a healthy weight through a healthy diet and exercise plan.

Healthy eating:

Your diabetes diet is simply a healthy-eating plan that will help you control your blood sugar.
You'll need to focus your diet on more fruits, vegetables, lean proteins and whole grains.
These are foods that are high in nutrition and fiber and low in fat and calories.
You'll also cut down on saturated fats, refined carbohydrates and sweets. In fact, it's the best eating plan for the entire family.
Sugary foods are OK once in a while. They must be counted as part of your meal plan.

Physical activity:

Everyone needs regular aerobic activity. This includes people who have diabetes.
Physical activity lowers your blood sugar level by moving sugar into your cells, where it's used for energy.
Physical activity also makes your body more sensitive to insulin. That means your body needs less insulin to transport sugar to your cells.

""")
    




                                                                                    
