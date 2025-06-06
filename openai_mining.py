import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai

# -------------------------------------------------------------------
# CONFIGURATION DE LA PAGE STREAMLIT
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Tableau de Bord d'Analyse de Similarité",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# CSS PERSONNALISÉ POUR LE STYLE
# -------------------------------------------------------------------
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------------------------------------------
@st.cache_data
def charger_donnees():
    """Charger et prétraiter les données de similarité"""
    try:
        # Assurez-vous que le fichier existe et c'est bien un .xlsx valide
        df = pd.read_excel("project_level_similarity_mixed.xlsx", engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

df = charger_donnees()
if df is None:
    st.stop()

# -------------------------------------------------------------------
# BARRE LATÉRALE - CONTRÔLES COMMUNS
# -------------------------------------------------------------------
st.sidebar.header("🎛️ Contrôles")

# Sliders pour les seuils TF-IDF et BERT
seuil_tfidf = st.sidebar.slider(
    "Seuil de Score TF-IDF",
    min_value=0.0,
    max_value=float(df['tfidf_score'].max()),
    value=0.0,
    step=0.01,
    help="Filtrer les projets dont le score TF-IDF est supérieur à ce seuil"
)

seuil_bert = st.sidebar.slider(
    "Seuil de Score BERT",
    min_value=float(df['bert_score'].min()),
    max_value=float(df['bert_score'].max()),
    value=float(df['bert_score'].min()),
    step=0.01,
    help="Filtrer les projets dont le score BERT est supérieur à ce seuil"
)

# Options d'analyse
afficher_statistiques = st.sidebar.checkbox("Afficher les Statistiques Détaillées", value=True)
afficher_top = st.sidebar.checkbox("Afficher les Meilleures Correspondances", value=True)
n_top_choisi = st.sidebar.selectbox("Nombre de Meilleures Correspondances", [5, 10, 15, 20], index=1)

# Appliquer les filtres EN DEHORS des onglets pour que df_filtre soit accessible partout
df_filtre = df[
    (df['tfidf_score'] >= seuil_tfidf) & 
    (df['bert_score'] >= seuil_bert)
]

# -------------------------------------------------------------------
# FONCTIONS UTILES
# -------------------------------------------------------------------
def creer_graphique_distribution(df_subset):
    """Créer des graphiques de distribution pour les deux méthodes de similarité"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribution des scores TF-IDF',
            'Distribution des scores BERT', 
            'TF-IDF vs BERT', 
            'Boîtes à moustaches'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # Histogramme TF-IDF
    fig.add_trace(
        go.Histogram(
            x=df_subset['tfidf_score'],
            name='TF-IDF',
            opacity=0.7,
            nbinsx=50,
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )

    # Histogramme BERT
    fig.add_trace(
        go.Histogram(
            x=df_subset['bert_score'],
            name='BERT',
            opacity=0.7,
            nbinsx=50,
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )

    # Nuage de points TF-IDF vs BERT
    fig.add_trace(
        go.Scatter(
            x=df_subset['tfidf_score'],
            y=df_subset['bert_score'],
            mode='markers',
            name='Projets',
            marker=dict(size=4, opacity=0.6),
            text=df_subset['acronyme_anr'] + ' - ' + df_subset['acronyme_cordis'],
            hovertemplate='<b>%{text}</b><br>TF-IDF : %{x:.4f}<br>BERT : %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Boîte à moustaches TF-IDF
    fig.add_trace(
        go.Box(
            y=df_subset['tfidf_score'],
            name='TF-IDF',
            marker_color='#1f77b4'
        ),
        row=2, col=2
    )

    # Boîte à moustaches BERT
    fig.add_trace(
        go.Box(
            y=df_subset['bert_score'],
            name='BERT',
            marker_color='#ff7f0e'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Analyse des Scores de Similarité",
        title_x=0.5
    )

    fig.update_xaxes(title_text="Score TF-IDF", row=1, col=1)
    fig.update_xaxes(title_text="Score BERT", row=1, col=2)
    fig.update_xaxes(title_text="Score TF-IDF", row=2, col=1)
    fig.update_xaxes(title_text="Méthodes", row=2, col=2)

    fig.update_yaxes(title_text="Fréquence", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=2)
    fig.update_yaxes(title_text="Score BERT", row=2, col=1)
    fig.update_yaxes(title_text="Valeur du Score", row=2, col=2)

    return fig

def creer_tableau_top(df_subset, n_top=10):
    """Créer un DataFrame des meilleures correspondances TF-IDF et BERT"""
    top_tfidf = df_subset.nlargest(n_top, 'tfidf_score')[
        ['acronyme_anr', 'acronyme_cordis', 'tfidf_score', 'bert_score']
    ].copy()
    top_tfidf['rang'] = range(1, len(top_tfidf) + 1)

    top_bert = df_subset.nlargest(n_top, 'bert_score')[
        ['acronyme_anr', 'acronyme_cordis', 'tfidf_score', 'bert_score']
    ].copy()
    top_bert['rang'] = range(1, len(top_bert) + 1)

    return top_tfidf, top_bert

# -------------------------------------------------------------------
# DÉFINITION DES ONGLETS
# -------------------------------------------------------------------
onglet1, onglet2, onglet3 = st.tabs([
    "🧮 Analyse de Similarité",
    "📊 Données",
    "💬 Chat & Comparaison de Résumés"
])

# -------------------------------------------------------------------
# ONGLET 1 : ANALYSE DE SIMILARITÉ
# -------------------------------------------------------------------
with onglet1:
    st.title("🔬 Tableau de Bord d'Analyse de Similarité de Projets")
    st.markdown("""
    <div class="highlight-box">
    Ce tableau de bord analyse les scores de similarité entre les projets ANR 
    (Agence Nationale de la Recherche) et CORDIS (projets de recherche européens) 
    en utilisant deux méthodes : TF-IDF et BERT.
    Utilisez les contrôles dans la barre latérale pour filtrer les données et explorer les différentes analyses.
    </div>
    """, unsafe_allow_html=True)

    # Indicateurs principaux
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Projets Totaux",
            f"{len(df_filtre):,}",
            delta=f"{len(df_filtre) - len(df):,}" if len(df_filtre) != len(df) else None
        )

    with col2:
        st.metric(
            "Moyenne TF-IDF",
            f"{df_filtre['tfidf_score'].mean():.4f}",
            delta=f"{(df_filtre['tfidf_score'].mean() - df['tfidf_score'].mean()):.4f}"
        )

    with col3:
        st.metric(
            "Moyenne BERT",
            f"{df_filtre['bert_score'].mean():.4f}",
            delta=f"{(df_filtre['bert_score'].mean() - df['bert_score'].mean()):.4f}"
        )

    with col4:
        correlation = df_filtre['tfidf_score'].corr(df_filtre['bert_score'])
        st.metric(
            "Corrélation des Scores",
            f"{correlation:.4f}",
            help="Corrélation de Pearson entre les scores TF-IDF et BERT"
        )

    # Visualisation principale
    st.header("📊 Distributions des Scores")
    fig = creer_graphique_distribution(df_filtre)
    st.plotly_chart(fig, use_container_width=True)

    # Statistiques détaillées
    if afficher_statistiques:
        st.header("📈 Statistiques Détaillées")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Statistiques TF-IDF")
            stats_tfidf = df_filtre['tfidf_score'].describe()
            st.dataframe(stats_tfidf.to_frame().T, use_container_width=True)

        with col2:
            st.subheader("Statistiques BERT")
            stats_bert = df_filtre['bert_score'].describe()
            st.dataframe(stats_bert.to_frame().T, use_container_width=True)

    # Meilleures correspondances
    if afficher_top:
        st.header("🏆 Meilleures Correspondances")

        top_tfidf, top_bert = creer_tableau_top(df_filtre, n_top_choisi)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top TF-IDF")
            st.dataframe(
                top_tfidf[['rang', 'acronyme_anr', 'acronyme_cordis', 'tfidf_score', 'bert_score']],
                column_config={
                    'rang': 'Rang',
                    'acronyme_anr': 'Projet ANR',
                    'acronyme_cordis': 'Projet CORDIS',
                    'tfidf_score': st.column_config.NumberColumn('Score TF-IDF', format="%.4f"),
                    'bert_score': st.column_config.NumberColumn('Score BERT', format="%.4f")
                },
                hide_index=True,
                use_container_width=True
            )

        with col2:
            st.subheader("Top BERT")
            st.dataframe(
                top_bert[['rang', 'acronyme_anr', 'acronyme_cordis', 'tfidf_score', 'bert_score']],
                column_config={
                    'rang': 'Rang',
                    'acronyme_anr': 'Projet ANR',
                    'acronyme_cordis': 'Projet CORDIS',
                    'tfidf_score': st.column_config.NumberColumn('Score TF-IDF', format="%.4f"),
                    'bert_score': st.column_config.NumberColumn('Score BERT', format="%.4f")
                },
                hide_index=True,
                use_container_width=True
            )

    # Comparaison des méthodes (metrics supplémentaires)
    st.header("🔍 Comparaison des Méthodes")
    col1, col2, col3 = st.columns(3)

    with col1:
        nb_tfidf_eleve = len(df_filtre[df_filtre['tfidf_score'] > 0.5])
        st.metric("Scores TF-IDF Élevés (>0.5)", nb_tfidf_eleve)

    with col2:
        nb_bert_eleve = len(df_filtre[df_filtre['bert_score'] > 0.5])
        st.metric("Scores BERT Élevés (>0.5)", nb_bert_eleve)

    with col3:
        nb_both_eleve = len(df_filtre[
            (df_filtre['tfidf_score'] > 0.5) & (df_filtre['bert_score'] > 0.5)
        ])
        st.metric("Scores Élevés dans les Deux (>0.5)", nb_both_eleve)

    # Carte de chaleur de corrélation
    st.subheader("Analyse de Corrélation des Scores")
    matrice_corr = df_filtre[['tfidf_score', 'bert_score']].corr()
    fig_corr = px.imshow(
        matrice_corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Matrice de Corrélation : TF-IDF vs BERT"
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Informations complémentaires
    st.header("💡 Informations Clés")
    insights = []
    total_projets = len(df_filtre)
    haute_agreement = len(df_filtre[
        (df_filtre['tfidf_score'] > 0.3) & (df_filtre['bert_score'] > 0.3)
    ])
    taux_agreement = (haute_agreement / total_projets * 100) if total_projets > 0 else 0

    insights.append(f"• **Accord des Méthodes** : {taux_agreement:.1f}% des paires de projets montrent une similarité élevée (>0.3) avec les deux méthodes")

    mediane_tfidf = df_filtre['tfidf_score'].median()
    mediane_bert = df_filtre['bert_score'].median()

    if mediane_tfidf > mediane_bert:
        insights.append(f"• **Distribution des Scores** : Les scores TF-IDF ont tendance à être plus élevés (médiane : {mediane_tfidf:.4f}) comparés aux scores BERT (médiane : {mediane_bert:.4f})")
    else:
        insights.append(f"• **Distribution des Scores** : Les scores BERT ont tendance à être plus élevés (médiane : {mediane_bert:.4f}) comparés aux scores TF-IDF (médiane : {mediane_tfidf:.4f})")

    if correlation > 0.5:
        insights.append(f"• **Fort niveau de Corrélation** : Les méthodes montrent une forte corrélation positive ({correlation:.3f}), indiquant une cohérence dans l'évaluation")
    elif correlation > 0.3:
        insights.append(f"• **Corrélation Modérée** : Les méthodes montrent une corrélation modérée ({correlation:.3f}), suggérant une certaine cohérence mais aussi des perspectives complémentaires")
    else:
        insights.append(f"• **Faible Corrélation** : Les méthodes montrent une faible corrélation ({correlation:.3f}), indiquant qu'elles capturent différents aspects de similarité")

    for info in insights:
        st.markdown(info)

    # Export des données filtrées
    st.header("💾 Export des Données Filtrées")
    if st.button("Télécharger les Données Filtrées au format CSV"):
        csv_data = df_filtre.to_csv(index=False)
        st.download_button(
            label="Télécharger le CSV",
            data=csv_data,
            file_name=f"donnees_filtrees_{len(df_filtre)}_projets.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------------
# ONGLET 2 : AFFICHAGE DE LA BASE COMPLÈTE
# -------------------------------------------------------------------
with onglet2:
    st.title("📊 Données Complètes")
    st.markdown("Cet onglet affiche **toutes** les lignes et colonnes du DataFrame original (sans filtrage).")

    # Affiche la base complète
    st.dataframe(df, use_container_width=True)

    # Statistiques numériques globales (describe) pour le DataFrame complet
    st.header("📈 Statistiques Numériques Globales")
    st.write(df.select_dtypes(include=[np.number]).describe().T)

    # Bouton pour télécharger le CSV complet
    csv_complet = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger le CSV Complet",
        data=csv_complet,
        file_name="cordis_full_dataset.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------------
# ONGLET 3 : CHAT & COMPARAISON DE RÉSUMÉS
# -------------------------------------------------------------------
with onglet3:
    st.title("💬 Chat et Comparaison de Résumés")
    st.markdown("""
    Dans cet onglet, vous pouvez fournir deux résumés et demander à l'assistant OpenAI 
    de les comparer. Un pré-prompt définit le rôle de l'assistant pour retourner uniquement 
    « Similaires » ou « Différents ».
    """)

    # Récupération de la clé OpenAI depuis les Secrets (vérifiez bien que le nom du secret est EXACTEMENT "openai_api_key")
    try:
        openai.api_key = st.secrets["openai_api_key"]
    except KeyError:
        st.error("⚠️ Clé OpenAI introuvable dans st.secrets. Nom attendu : 'openai_api_key'.")

    # Pré-prompt minimal pour ne retourner que "Similaires" ou "Différents"
    pre_prompt = (
        "Vous êtes un assistant dont la seule tâche est de comparer deux résumés "
        "et de retourner exactement 'Similaires' ou 'Différents'. "
        "Si le contenu des deux résumés est essentiellement le même (mêmes idées, mêmes points clés), "
        "répondez strictement par « Similaires ». Sinon, répondez strictement par « Différents ». "
        "Aucun mot supplémentaire, aucune explication."
    )

    # Champs de texte pour les deux résumés
    resume1 = st.text_area("Résumé 1", height=200, placeholder="Collez le premier résumé ici...")
    resume2 = st.text_area("Résumé 2", height=200, placeholder="Collez le deuxième résumé ici...")

    # Bouton pour lancer la comparaison
    if st.button("Comparer les Résumés"):
        if not resume1.strip() or not resume2.strip():
            st.error("Veuillez remplir les deux champs de résumé avant de comparer.")
        else:
            messages = [
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": f"Voici le premier résumé :\n\n{resume1}"},
                {"role": "user", "content": f"Voici le deuxième résumé :\n\n{resume2}"}
            ]

            try:
                with st.spinner("Analyse en cours…"):
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",   # ou un autre modèle selon votre abonnement
                        messages=messages,
                        temperature=0.0,
                        max_tokens=5
                    )
                label = response.choices[0].message.content.strip()
                st.subheader("Résultat :")
                st.write(f"**{label}**")
            except Exception as e:
                st.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")

# -------------------------------------------------------------------
# FIN DU SCRIPT
# -------------------------------------------------------------------
