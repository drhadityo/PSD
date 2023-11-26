import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('Agg')

@st.cache
def load_data(dataset):
  df = pd.read_csv(dataset)
  return df

def load_predict_model(model_file):
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

buying_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
maint_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
doors_label = {'2': 0, '3': 1, '5more': 2, '4': 3}
persons_label = {'2': 0, '4': 1, 'more': 2}
lug_boot_label = {'small': 0, 'big': 1, 'med': 2}
safety_label = {'high': 0, 'med': 1, 'low': 2}
class_label = {'good': 0, 'acceptable': 1, 'very good': 2, 'unacceptable': 3}

# Get the Keys
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# Find the Key From Dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Set page title and icon
st.set_page_config(
    page_title="Car Evaluation App",
    page_icon="🚗",
    layout="wide"  # Mengatur layout ke mode wide
)

def main():
    st.title("Car Evaluation App")
    st.subheader("Dohan Rizqi Hadityo | 210411100195")

    # Menu
    menu = ["EDA", "Prediction"]
    choice = st.sidebar.selectbox("Pilih Menu", menu)

    if choice == 'EDA':
        st.subheader("EDA")

        my_data = load_data('data/car_data_6fitur.csv')
        st.dataframe(my_data.head(15))

        if st.checkbox("Show Summary"):
            st.write(my_data.describe())

        if st.checkbox("Show Shape"):
            st.write(my_data.shape)

        if st.checkbox("Value Count Plot"):
            fig, ax = plt.subplots()
            my_data['class'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        if st.checkbox("Pie Chart"):
            fig, ax = plt.subplots()
            my_data['class'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)

    if choice == 'Prediction':
        st.subheader("Prediksi")

        buying = st.selectbox("Select Buying Level", tuple(buying_label.keys()))
        maint = st.selectbox("Select Maintenance Level", tuple(maint_label.keys()))
        doors = st.selectbox("Select Doors", tuple(doors_label.keys()))
        persons = st.number_input("Select Num of Persons", 2, 10)
        lug_boot = st.selectbox("Select Lug Boot", tuple(lug_boot_label.keys()))
        safety = st.selectbox("Select Safety", tuple(safety_label.keys()))

        # Encoding
        v_buying = get_value(buying, buying_label)
        v_maint = get_value(maint, maint_label)
        v_doors = get_value(doors, doors_label)
        v_lug_boot = get_value(lug_boot, lug_boot_label)
        v_safety = get_value(safety, safety_label)

        hasil_input_data = {
            "buying": buying,
            "maint": maint,
            "doors": doors,
            "persons": persons,
            "lug_boot": lug_boot,
            "safety": safety,
        }
        st.subheader("Opsi Di Input")
        st.json(hasil_input_data)

        st.subheader("Data di Encoding Sebagai")
        # Data To Be Used
        input_data = [v_buying, v_maint, v_doors, persons, v_lug_boot, v_safety]
        st.write(input_data)

        prep_data = np.array(input_data).reshape(1, -1)

        # Tambahkan tombol untuk melakukan prediksi
        if st.button('Prediksi Kelas'):
            # Hanya menggunakan model Random Forest
            predictor = load_predict_model('pickle/randomforest_car_model.pkl')
            prediction = predictor.predict(prep_data)

            final_result = get_key(prediction, class_label)
            st.success(final_result)

if __name__ == '__main__':
    main()
