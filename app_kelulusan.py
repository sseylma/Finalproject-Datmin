import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="🎓 Prediksi Kelulusan Mahasiswa", layout="wide")

# --- Sidebar ---
st.sidebar.title("🎛️ Pengaturan Aplikasi")
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset CSV", type=["csv"])
algorithm = st.sidebar.selectbox("🧠 Pilih Algoritma", 
    ["Regresi Linier", "Regresi Logistik", "Naive Bayes", "SVM", "K-NN", "Decision Tree", "K-Means"]
)
run_model = st.sidebar.button("🚀 Jalankan")

# --- Main UI ---
st.markdown("# 🎓 Prediksi Tingkat Kelulusan Mahasiswa")
st.markdown("Gunakan algoritma data mining untuk memprediksi kelulusan berdasarkan faktor akademik dan sosial.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("## 👀 Preview Dataset")
    st.dataframe(df.head())

    st.markdown("## ⚙️ Pilih Kolom Target")
    target_column = st.selectbox("🎯 Pilih Kolom Target (jika diperlukan)", options=[None] + list(df.columns))

    if run_model:
        st.markdown("## 🚧 Hasil Prediksi & Evaluasi")

        if algorithm == "K-Means":
            features = df.select_dtypes(include=["int64", "float64"])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df["Cluster"] = clusters

            st.success("✅ Clustering selesai. Lihat hasil di bawah.")
            st.write(df[["Cluster"] + list(features.columns)].head())

            st.markdown("### 📊 Visualisasi Cluster (PCA/2D)")
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                fig, ax = plt.subplots()
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis", ax=ax)
                st.pyplot(fig)
            except:
                st.warning("Gagal visualisasi PCA. Coba dataset numerik saja.")
        else:
            if target_column is None:
                st.warning("❗ Pilih kolom target terlebih dahulu untuk algoritma supervised.")
            else:
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Encode jika ada kategori
                for col in X.select_dtypes(include=["object"]).columns:
                    X[col] = LabelEncoder().fit_transform(X[col])
                if y.dtype == "object":
                    y = LabelEncoder().fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = None
                if algorithm == "Regresi Linier":
                    model = LinearRegression()
                elif algorithm == "Regresi Logistik":
                    model = LogisticRegression(max_iter=1000)
                elif algorithm == "Naive Bayes":
                    model = GaussianNB()
                elif algorithm == "SVM":
                    model = SVC()
                elif algorithm == "K-NN":
                    model = KNeighborsClassifier()
                elif algorithm == "Decision Tree":
                    model = DecisionTreeClassifier()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.success(f"🎯 Akurasi Model: **{acc * 100:.2f}%**")

                st.markdown("### 🔍 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.markdown("### 📝 Classification Report")
                st.text(classification_report(y_test, y_pred))
