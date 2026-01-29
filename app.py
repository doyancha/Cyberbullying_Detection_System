import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.set_page_config(
    page_title="Cyberbullying Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic styling
st.markdown("""
<style>
  .main-header {background: linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:10px;color:#fff;text-align:center}
  .metric-card{background:#fff;padding:16px;border-radius:8px}
</style>
""", unsafe_allow_html=True)

# Session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

@st.cache_data
def load_dataset():
    path = 'aggression_parsed_dataset.csv'
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

df = load_dataset()

@st.cache_resource
def load_test_split():
    """Load test split saved from notebook (X_test, y_test)"""
    import scipy.sparse as sp
    X_test_path = 'X_test_sparse.npz'
    y_test_path = 'y_test.npy'
    try:
        if os.path.exists(X_test_path) and os.path.exists(y_test_path):
            X_test = sp.load_npz(X_test_path)
            y_test = np.load(y_test_path)
            return X_test, y_test
    except Exception:
        pass
    return None, None

X_test_loaded, y_test_loaded = load_test_split()

# Text preprocessing used by the UI
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '<user>', text)
    text = re.sub(r'http\S+|www\S+', '<url>', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Simple simulated predictor used when a real model isn't available
def predict_simulated(text):
    bullying_keywords = ['stupid', 'idiot', 'dumb', 'hate', 'kill', 'die',
                         'loser', 'worthless', 'ugly', 'fat', 'nobody']
    t = str(text).lower()
    keyword_count = sum(1 for w in bullying_keywords if w in t)
    if keyword_count >= 2:
        confidence = min(0.95, 0.65 + (keyword_count * 0.1))
        pred = 1
    elif keyword_count == 1:
        confidence = 0.55 + (len(t.split()) * 0.01)
        pred = 1 if confidence > 0.6 else 0
    else:
        confidence = max(0.15, 0.35 - (len(t.split()) * 0.01))
        pred = 0
    triggered = [w for w in bullying_keywords if w in t]
    return int(pred), float(confidence), triggered


# Wrapper that uses loaded model/vectorizer if available, else simulated
def predict(text):
    if 'model' in globals() and 'vectorizer' in globals() and model is not None and vectorizer is not None:
        try:
            clean = preprocess_text(text)
            X = vectorizer.transform([clean])
            pred = int(model.predict(X)[0])
            prob = None
            if hasattr(model, 'predict_proba'):
                try:
                    prob = float(model.predict_proba(X)[0][1])
                except Exception:
                    prob = None
            # extract some simple triggered words from text
            keywords = ['stupid','idiot','dumb','hate','kill','die','loser','worthless','ugly','fat','nobody']
            triggered = [w for w in keywords if w in str(text).lower()]
            return int(pred), (prob if prob is not None else (0.9 if pred == 1 else 0.1)), triggered
        except Exception:
            return predict_simulated(text)
    else:
        return predict_simulated(text)

# Sidebar controls already rendered later
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/security-shield-green.png", width=100)
    st.title("Navigation")
    page = st.radio("Select Page:", ["🏠 Live Detection", "📊 Analytics Dashboard", "⚙️ System Info", "📚 About"]) 
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Model Loaded", "Yes" if 'model' in globals() and model is not None and 'vectorizer' in globals() and vectorizer is not None else "No")
    st.metric("Analyses Today", len(st.session_state.analysis_history))
    st.markdown("---")
    st.subheader("Optional uploads")
    uploaded_model = st.file_uploader("Upload model (.pkl)", type=['pkl'])
    uploaded_vectorizer = st.file_uploader("Upload vectorizer (.pkl)", type=['pkl'])
    if uploaded_model is not None or uploaded_vectorizer is not None:
        try:
            if uploaded_model is not None:
                model = joblib.load(uploaded_model)
            if uploaded_vectorizer is not None:
                vectorizer = joblib.load(uploaded_vectorizer)
            st.success("Uploaded model/vectorizer loaded (temporary).")
        except Exception:
            st.error("Failed to load uploaded file(s). Make sure they're valid joblib pickle files.")
    st.markdown("---")
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    uploaded_dataset = st.file_uploader("Upload dataset CSV for analytics (optional)", type=['csv'])

st.markdown("""
<div class="main-header">
    <h1>🛡️ Cyberbullying Detection Dashboard</h1>
    <p>AI-Powered Content Moderation System</p>
</div>
""", unsafe_allow_html=True)

# Page: Live Detection
if page == "🏠 Live Detection":
    st.header("🔍 Real-Time Cyberbullying Detection")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Enter Text to Analyze")
        input_method = st.radio("Input method:", ["Type text","Try examples"]) 
        if input_method == "Type text":
            user_text = st.text_area("Enter the text you want to analyze:", height=150)
        else:
            examples = {
                "Severe Cyberbullying":"You're so stupid, nobody likes you and you should just kill yourself",
                "Moderate Harassment":"lol you're such a loser haha everyone thinks you're dumb",
                "Safe Content":"Great job on the presentation today! Well done!"
            }
            sel = st.selectbox("Choose an example:", list(examples.keys()))
            user_text = examples[sel]
            st.text_area("Selected text:", user_text, height=100)

        if st.button("🔍 Analyze Text") and user_text:
            with st.spinner("Analyzing..."):
                cleaned = preprocess_text(user_text)
                pred, conf, triggered = predict(user_text)
                decision = 1 if conf >= threshold else 0
                st.session_state.analysis_history.append({'timestamp':datetime.now(),'text':user_text[:200],'prediction':int(decision),'confidence':float(conf)})
                st.markdown("---")
                st.subheader("📋 Analysis Results")
                if decision==1:
                    if conf>=0.8:
                        st.markdown(f"""
                        <div class='alert-high'><h3>⚠️ HIGH RISK</h3><p><strong>Confidence: {conf*100:.1f}%</strong></p></div>
                        """, unsafe_allow_html=True)
                        rc='red'
                    elif conf>=0.6:
                        st.markdown(f"""
                        <div class='alert-medium'><h3>⚠️ MODERATE RISK</h3><p><strong>Confidence: {conf*100:.1f}%</strong></p></div>
                        """, unsafe_allow_html=True)
                        rc='orange'
                    else:
                        st.markdown(f"""
                        <div class='alert-low'><h3>⚠️ LOW RISK</h3><p><strong>Confidence: {conf*100:.1f}%</strong></p></div>
                        """, unsafe_allow_html=True)
                        rc='yellow'
                else:
                    st.markdown(f"""
                    <div class='safe-content'><h3>✅ SAFE CONTENT</h3><p><strong>Confidence: {(1-conf)*100:.1f}%</strong></p></div>
                    """, unsafe_allow_html=True)
                    rc='green'
                val = conf*100 if decision==1 else (1-conf)*100
                fig = go.Figure(go.Indicator(mode='gauge+number',value=val,title={'text':'Confidence Level'},gauge={'axis':{'range':[0,100]},'bar':{'color':rc}}))
                fig.update_layout(height=250)
                st.plotly_chart(fig,use_container_width=True)
                if triggered:
                    st.markdown('### 🎯 Flagged Terms')
                    st.warning(f"Detected words: {', '.join(triggered)}")
                st.markdown('### 💡 Recommendation')
                if decision==1 and conf>=0.8:
                    st.error('Action required: human review')
                elif decision==1 and conf>=0.6:
                    st.warning('Manual review recommended')
                elif decision==1:
                    st.info('Monitor this user')
                else:
                    st.success('No action needed')

    with col2:
        st.subheader('📊 Analysis Metrics')
        metrics = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1-Score'],'Value':[81.0,96.5,80.2,87.6]})
        fig = px.bar(metrics,x='Metric',y='Value',color='Value',color_continuous_scale='RdYlGn',text='Value')
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        fig.update_layout(showlegend=False,height=300,yaxis_range=[0,100])
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('---')
        st.subheader('📜 Recent Analyses')
        if st.session_state.analysis_history:
            recent = st.session_state.analysis_history[-5:][::-1]
            for it in recent:
                status = '🔴' if it['prediction']==1 else '🟢'
                ts = it['timestamp'].strftime('%H:%M:%S') if isinstance(it['timestamp'],datetime) else str(it['timestamp'])
                st.markdown(f"**{status} {ts}**  \n*{it['text'][:80]}*  \nConfidence: {it['confidence']*100:.1f}%")
                st.markdown('---')
        else:
            st.info('No analyses yet')

elif page == '📊 Analytics Dashboard':
    st.header('📊 System Analytics & Performance')
    # use uploaded dataset if present
    if 'uploaded_dataset' in locals() and uploaded_dataset is not None:
        try:
            df_use = pd.read_csv(uploaded_dataset)
        except Exception:
            df_use = df
    else:
        df_use = df

    # If model, vectorizer, and test split are present, compute metrics on test set
    if 'model' in globals() and 'vectorizer' in globals() and model is not None and vectorizer is not None and X_test_loaded is not None and y_test_loaded is not None:
        try:
            y_pred = model.predict(X_test_loaded)
            y_true = y_test_loaded
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test_loaded)[:, 1]
                except Exception:
                    y_proba = None

            acc = accuracy_score(y_true, y_pred) * 100
            prec = precision_score(y_true, y_pred, zero_division=0) * 100
            rec = recall_score(y_true, y_pred, zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, zero_division=0) * 100

            # Top features from model coefficients
            top_features_df = None
            if hasattr(model, 'coef_') and hasattr(vectorizer, 'get_feature_names_out'):
                try:
                    feat_names = vectorizer.get_feature_names_out()
                    coefs = model.coef_[0]
                    top_idx = np.argsort(coefs)[-15:][::-1]
                    top_feats = [(feat_names[i], float(coefs[i])) for i in top_idx]
                    top_features_df = pd.DataFrame(top_feats, columns=['Word','Weight'])
                except Exception:
                    top_features_df = None

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Render metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f"""<div class='metric-card'><h2 style='color:#667eea'>{acc:.1f}%</h2><p>Accuracy</p></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class='metric-card'><h2 style='color:#764ba2'>{prec:.1f}%</h2><p>Precision</p></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class='metric-card'><h2 style='color:#f093fb'>{rec:.1f}%</h2><p>Recall</p></div>""", unsafe_allow_html=True)
            c4.markdown(f"""<div class='metric-card'><h2 style='color:#4facfe'>{f1:.1f}%</h2><p>F1-Score</p></div>""", unsafe_allow_html=True)
            st.info(f"📊 Metrics computed on test set ({X_test_loaded.shape[0]} samples)")

            st.markdown('---')
            r1,r2 = st.columns(2)
            with r1:
                st.subheader('🔲 Confusion Matrix (Test Set)')
                cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
                fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='RdYlGn_r')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            with r2:
                st.subheader('🥧 Test Set Distribution (by label)')
                dist_data = pd.Series(y_test_loaded).value_counts().rename_axis('label').reset_index(name='count')
                dist_data['label'] = dist_data['label'].astype(str)
                fig = px.pie(dist_data, values='count', names='label', color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

            st.markdown('---')
            t1,t2 = st.columns(2)
            with t1:
                st.subheader('📈 Top Cyberbullying Indicators (from model)')
                if top_features_df is not None and not top_features_df.empty:
                    fig = px.bar(top_features_df.head(10), y='Word', x='Weight', orientation='h', color='Weight', color_continuous_scale='Reds')
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info('No feature importance available for the loaded model.')
            with t2:
                st.subheader('📉 Top Safe Content Indicators')
                if top_features_df is not None and not top_features_df.empty:
                    # take bottom-weighted features as 'safe' indicators
                    safe_df = top_features_df.tail(10).sort_values('Weight', ascending=False)
                    fig = px.bar(safe_df, y='Word', x='Weight', orientation='h', color='Weight', color_continuous_scale='Greens')
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info('No safe indicator data available.')

        except Exception as e:
            st.error(f"Could not compute analytics from test split: {e}")
    # Fallback: try to use full dataset if no test split available
    elif 'model' in globals() and 'vectorizer' in globals() and model is not None and vectorizer is not None and not df_use.empty:
        # ensure Text and oh_label columns exist
        if 'Text' in df_use.columns and 'oh_label' in df_use.columns:
            st.warning("Test split not found. Computing metrics on full dataset (may differ from notebook results).")
            df_eval = df_use.copy()
            # clean text using NLTK-style cleaning similar to the notebook
            def clean_text_nltk(text):
                txt = str(text).lower()
                txt = re.sub(r'http\\S+|www\\S+', '', txt)
                txt = re.sub(r'@\\w+', '', txt)
                txt = re.sub(r'[^a-z\\s]', ' ', txt)
                txt = ' '.join([lemmatizer.lemmatize(w) for w in txt.split() if w not in stop_words])
                return txt.strip()

            df_eval['cleaned_text'] = df_eval['Text'].apply(clean_text_nltk)

            # Transform and predict
            try:
                X_all = vectorizer.transform(df_eval['cleaned_text'])
                y_true = df_eval['oh_label'].astype(int).values
                y_pred = model.predict(X_all)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_all)[:, 1]
                    except Exception:
                        y_proba = None

                acc = accuracy_score(y_true, y_pred) * 100
                prec = precision_score(y_true, y_pred, zero_division=0) * 100
                rec = recall_score(y_true, y_pred, zero_division=0) * 100
                f1 = f1_score(y_true, y_pred, zero_division=0) * 100

                # Top features from model coefficients
                top_features_df = None
                if hasattr(model, 'coef_') and hasattr(vectorizer, 'get_feature_names_out'):
                    try:
                        feat_names = vectorizer.get_feature_names_out()
                        coefs = model.coef_[0]
                        top_idx = np.argsort(coefs)[-15:][::-1]
                        top_feats = [(feat_names[i], float(coefs[i])) for i in top_idx]
                        top_features_df = pd.DataFrame(top_feats, columns=['Word','Weight'])
                    except Exception:
                        top_features_df = None

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                # Render metrics
                c1,c2,c3,c4 = st.columns(4)
                c1.markdown(f"""<div class='metric-card'><h2 style='color:#667eea'>{acc:.1f}%</h2><p>Accuracy</p></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class='metric-card'><h2 style='color:#764ba2'>{prec:.1f}%</h2><p>Precision</p></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class='metric-card'><h2 style='color:#f093fb'>{rec:.1f}%</h2><p>Recall</p></div>""", unsafe_allow_html=True)
                c4.markdown(f"""<div class='metric-card'><h2 style='color:#4facfe'>{f1:.1f}%</h2><p>F1-Score</p></div>""", unsafe_allow_html=True)

                st.markdown('---')
                r1,r2 = st.columns(2)
                with r1:
                    st.subheader('🔲 Confusion Matrix')
                    cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
                    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='RdYlGn_r')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                with r2:
                    st.subheader('🥧 Dataset Distribution (by label)')
                    dist = df_eval['oh_label'].value_counts().rename_axis('label').reset_index(name='count')
                    dist['label'] = dist['label'].astype(str)
                    fig = px.pie(dist, values='count', names='label', color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')

                st.markdown('---')
                t1,t2 = st.columns(2)
                with t1:
                    st.subheader('📈 Top Cyberbullying Indicators (from model)')
                    if top_features_df is not None and not top_features_df.empty:
                        fig = px.bar(top_features_df.head(10), y='Word', x='Weight', orientation='h', color='Weight', color_continuous_scale='Reds')
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info('No feature importance available for the loaded model.')
                with t2:
                    st.subheader('📉 Top Safe Content Indicators')
                    if top_features_df is not None and not top_features_df.empty:
                        # take bottom-weighted features as 'safe' indicators
                        safe_df = top_features_df.tail(10).sort_values('Weight', ascending=False)
                        fig = px.bar(safe_df, y='Word', x='Weight', orientation='h', color='Weight', color_continuous_scale='Greens')
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info('No safe indicator data available.')

            except Exception as e:
                st.error(f"Could not compute analytics from model/vectorizer: {e}")
        else:
            st.warning('Dataset must contain "Text" and "oh_label" columns to compute analytics. Showing placeholders.')
    else:
        # Placeholders when model/vectorizer or dataset unavailable
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown("""<div class='metric-card'><h2 style='color:#667eea'>81.0%</h2><p>Accuracy</p></div>""",unsafe_allow_html=True)
        c2.markdown("""<div class='metric-card'><h2 style='color:#764ba2'>96.5%</h2><p>Precision</p></div>""",unsafe_allow_html=True)
        c3.markdown("""<div class='metric-card'><h2 style='color:#f093fb'>80.2%</h2><p>Recall</p></div>""",unsafe_allow_html=True)
        c4.markdown("""<div class='metric-card'><h2 style='color:#4facfe'>87.6%</h2><p>F1-Score</p></div>""",unsafe_allow_html=True)
        st.markdown('---')
        r1,r2 = st.columns(2)
        with r1:
            st.subheader('🔲 Confusion Matrix')
            cm = pd.DataFrame({'Not CB (Predicted)':[1300,1548],'CB (Predicted)':[231,6287]},index=['Not CB (Actual)','CB (Actual)'])
            fig = px.imshow(cm,text_auto=True,color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=400)
            st.plotly_chart(fig,width='stretch')
        with r2:
            st.subheader('🥧 Dataset Distribution')
            class_data = pd.DataFrame({'Category':['Religion','Age','Gender','Ethnicity','Not CB','Other CB'],'Count':[7995,7988,7875,7955,7657,7358]})
            fig = px.pie(class_data,values='Count',names='Category',color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig,width='stretch')
        st.markdown('---')
        t1,t2 = st.columns(2)
        with t1:
            st.subheader('📈 Top Cyberbullying Indicators')
            bullying_words = pd.DataFrame({'Word':['rape','dumb','bullies','nigger','feminazi','idiot','muslims','idiots','gay','bitch'],'Weight':[8.38,8.26,8.10,8.10,7.87,6.69,6.40,6.37,6.06,5.91]})
            fig = px.bar(bullying_words,y='Word',x='Weight',orientation='h',color='Weight',color_continuous_scale='Reds')
            fig.update_layout(height=400,showlegend=False)
            st.plotly_chart(fig,width='stretch')
        with t2:
            st.subheader('📉 Top Safe Content Indicators')
            safe_words = pd.DataFrame({'Word':['mkr','daesh','class','mosul','bullying','college','yesallwomen','kat and','andre','user also'],'Weight':[5.39,3.43,3.42,3.40,3.23,2.70,2.45,2.38,2.35,2.17]})
            fig = px.bar(safe_words,y='Word',x='Weight',orientation='h',color='Weight',color_continuous_scale='Greens')
            fig.update_layout(height=400,showlegend=False)
            st.plotly_chart(fig,width='stretch')

elif page == '⚙️ System Info':
    st.header('⚙️ System Information')
    c1,c2 = st.columns(2)
    with c1:
        st.subheader('🤖 Model Details')
        st.markdown('''**Algorithm:** Logistic Regression  
**Feature Extraction:** TF-IDF  
**Vocabulary Size:** 5,000 features  
**Training Samples:** 37,462 tweets  
**Test Samples:** 9,366 tweets''')
    with c2:
        st.subheader('🎯 Performance Targets')
        targets = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1-Score'],'Target':[75,90,75,80],'Achieved':[81.0,96.5,80.2,87.6]})
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Target',x=targets['Metric'],y=targets['Target'],marker_color='lightgray'))
        fig.add_trace(go.Bar(name='Achieved',x=targets['Metric'],y=targets['Achieved'],marker_color='#667eea'))
        fig.update_layout(height=300,barmode='group')
        st.plotly_chart(fig,use_container_width=True)

else:
    st.header('📚 About This System')
    st.markdown('''This system uses AI and NLP to detect potentially harmful content. Use as a flagging tool; require human review for actions.''')

st.markdown('---')
st.markdown("""
<div style='text-align:center;color:gray;padding:1rem'>
  <strong>Cyberbullying Detection System v1.0</strong>
</div>
""", unsafe_allow_html=True)
