import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import time

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
        df = pd.read_excel("project_level_similarity_mixed.xlsx", engine="openpyxl")
        
        # Vérifier les colonnes requises
        colonnes_requises = ['tfidf_score', 'bert_score']
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        
        if colonnes_manquantes:
            st.error("Colonnes manquantes : {}".format(colonnes_manquantes))
            return None
        
        # Nettoyer les données
        df = df.dropna(subset=['tfidf_score', 'bert_score'])
        df['tfidf_score'] = pd.to_numeric(df['tfidf_score'], errors='coerce')
        df['bert_score'] = pd.to_numeric(df['bert_score'], errors='coerce')
        df = df.dropna(subset=['tfidf_score', 'bert_score'])
        
        st.success("Données chargées avec succès : {} enregistrements".format(len(df)))
        return df
        
    except Exception as e:
        st.error("Erreur lors du chargement des données : {}".format(e))
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
    if df_subset.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée ne correspond aux critères de filtrage actuels",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
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
    hover_text = []
    for _, row in df_subset.iterrows():
        anr_name = row.get('acronyme_anr', 'Inconnu')
        cordis_name = row.get('acronyme_cordis', 'Inconnu')
        hover_text.append(f"{anr_name} - {cordis_name}")
    
    fig.add_trace(
        go.Scatter(
            x=df_subset['tfidf_score'],
            y=df_subset['bert_score'],
            mode='markers',
            name='Projets',
            marker=dict(size=4, opacity=0.6),
            text=hover_text,
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
    if df_subset.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    top_tfidf = df_subset.nlargest(n_top, 'tfidf_score')[
        ['acronyme_anr', 'acronyme_cordis', 'tfidf_score', 'bert_score']
    ].copy()
    top_tfidf['rang'] = range(1, len(top_tfidf) + 1)

    top_bert = df_subset.nlargest(n_top, 'bert_score')[
        ['acronyme_anr', 'acronyme_cordis', 'tfidf_score', 'bert_score']
    ].copy()
    top_bert['rang'] = range(1, len(top_bert) + 1)

    return top_tfidf, top_bert

def comparer_resumes_avec_openai(resume1, resume2, api_key, type_comparaison="detailed"):
    """Compare deux résumés de projets via l'API OpenAI"""
    if type_comparaison == "detailed":
        prompt = """Vous êtes chargé de comparer deux projets de recherche pour un comité d'évaluation ANR/CORDIS.
Votre réponse doit être rigoureuse, structurée et actionnable. Analysez les deux résumés selon :

1. **Domaine scientifique et thématique** : Quelle discipline principale, quels champs secondaires ? Quelles similarités ou écarts sur les axes de recherche ?
2. **Objectifs et méthodologie** : Buts explicites, problématiques abordées, grandes lignes méthodologiques. Y a-t-il convergence, redondance, ou complémentarité ?
3. **Impact et retombées** : Quels résultats attendus, quelles applications concrètes ou valorisations ? Sont-ils compétitifs ou redondants par rapport au paysage existant ?
4. **Ressources, partenariats, consortium** : Indiquez les partenariats clés, le profil des équipes, les moyens mobilisés.
5. **Synthèse comparative et score** : Donnez un score de similarité globale (0–100 %), justifiez-le en trois lignes max, et terminez par une recommandation ("Fusionner", "Différencier", "Compléter", etc).

**Résumé Projet ANR** :
{}

**Résumé Projet CORDIS** :
{}

Structurez chaque partie, soyez synthétique et critique, puis finissez par le score et la recommandation.""".format(resume1, resume2)
    else:  # simple - oui/non
        prompt = """Vous êtes un assistant dont la seule tâche est de comparer deux résumés de projets de recherche.
Analysez si ces deux projets sont SIMILAIRES ou DIFFÉRENTS en termes de :
- Domaine de recherche
- Objectifs principaux  
- Méthodologie générale
- Applications visées

Répondez UNIQUEMENT par "SIMILAIRES" si les projets traitent essentiellement du même sujet avec des approches comparables.
Répondez UNIQUEMENT par "DIFFÉRENTS" dans tous les autres cas.
Aucun autre mot, aucune explication.

Résumé 1: {}
Résumé 2: {}""".format(resume1, resume2)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Vous êtes un expert en comparaison de projets de recherche."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000 if type_comparaison != "simple" else 5,
            temperature=0.3
        )
        return response.choices[0].message.content

    except AttributeError:
        # Pour les anciennes versions d'OpenAI
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Vous êtes un expert en comparaison de projets de recherche."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000 if type_comparaison != "simple" else 5,
            temperature=0.3
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return "Erreur lors de l'appel à l'API OpenAI : {}".format(str(e))

def extraire_trl_avec_openai(texte_resume, api_key):
    """Extrait l'échelle TRL d'un résumé de projet via l'API OpenAI"""
    prompt = """Tu es un expert en analyse de projets de recherche et développement.
Ton objectif est d'extraire une estimation de l'échelle TRL (Technology Readiness Level) de ce résumé.

Instructions :
1. Évalue le TRL sur une échelle de 1 à 9 (TRL 1 = recherche fondamentale, TRL 9 = système opérationnel)
2. Si impossible à déterminer, indique "TRL indéterminé"
3. Réponds uniquement par "TRL X" où X est un nombre de 1-9 ou "indéterminé"

Résumé du projet : "{}" """.format(texte_resume)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en évaluation TRL. Tu réponds uniquement par 'TRL X' ou 'TRL indéterminé'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    except AttributeError:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en évaluation TRL. Tu réponds uniquement par 'TRL X' ou 'TRL indéterminé'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return "Erreur lors de l'extraction TRL : {}".format(str(e))

# -------------------------------------------------------------------
# DÉFINITION DES ONGLETS
# -------------------------------------------------------------------
onglet1, onglet2, onglet3, onglet4 = st.tabs([
    "🧮 Analyse de Similarité",
    "📊 Données",
    "🎯 Extraction TRL",
    "💬 Comparaison de Résumés"
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

    # Vérifier s'il y a des données filtrées
    if df_filtre.empty:
        st.warning("⚠️ Aucune donnée ne correspond aux critères de filtrage actuels. Veuillez ajuster les seuils.")
    else:
        # Indicateurs principaux
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Projets Totaux",
                "{:,}".format(len(df_filtre)),
                delta="{:,}".format(len(df_filtre) - len(df)) if len(df_filtre) != len(df) else None
            )

        with col2:
            moyenne_tfidf = df_filtre['tfidf_score'].mean()
            delta_tfidf = moyenne_tfidf - df['tfidf_score'].mean()
            st.metric(
                "Moyenne TF-IDF",
                "{:.4f}".format(moyenne_tfidf),
                delta="{:.4f}".format(delta_tfidf) if len(df_filtre) != len(df) else None
            )

        with col3:
            moyenne_bert = df_filtre['bert_score'].mean()
            delta_bert = moyenne_bert - df['bert_score'].mean()
            st.metric(
                "Moyenne BERT",
                "{:.4f}".format(moyenne_bert),
                delta="{:.4f}".format(delta_bert) if len(df_filtre) != len(df) else None
            )

        with col4:
            if len(df_filtre) > 1:
                correlation = df_filtre['tfidf_score'].corr(df_filtre['bert_score'])
                st.metric(
                    "Corrélation des Scores",
                    "{:.4f}".format(correlation),
                    help="Corrélation de Pearson entre les scores TF-IDF et BERT"
                )
            else:
                st.metric("Corrélation des Scores", "N/A")

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
            
            if not top_tfidf.empty and not top_bert.empty:
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

        # Comparaison des méthodes
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

        # Informations clés (Insights)
        st.header("💡 Informations Clés")
        insights = []
        total_projets = len(df_filtre)
        haute_agreement = len(df_filtre[
            (df_filtre['tfidf_score'] > 0.3) & (df_filtre['bert_score'] > 0.3)
        ])
        taux_agreement = (haute_agreement / total_projets * 100) if total_projets > 0 else 0

        insights.append("• **Accord des Méthodes** : {:.1f}% des paires de projets montrent une similarité élevée (>0.3) avec les deux méthodes".format(taux_agreement))

        mediane_tfidf = df_filtre['tfidf_score'].median()
        mediane_bert = df_filtre['bert_score'].median()

        if mediane_tfidf > mediane_bert:
            insights.append("• **Distribution des Scores** : Les scores TF-IDF ont tendance à être plus élevés (médiane : {:.4f}) comparés aux scores BERT (médiane : {:.4f})".format(mediane_tfidf, mediane_bert))
        else:
            insights.append("• **Distribution des Scores** : Les scores BERT ont tendance à être plus élevés (médiane : {:.4f}) comparés aux scores TF-IDF (médiane : {:.4f})".format(mediane_bert, mediane_tfidf))

        if len(df_filtre) > 1:
            correlation = df_filtre['tfidf_score'].corr(df_filtre['bert_score'])
            if correlation > 0.5:
                insights.append("• **Forte Corrélation** : Les méthodes montrent une forte corrélation positive ({:.3f}), indiquant une cohérence dans l'évaluation".format(correlation))
            elif correlation > 0.3:
                insights.append("• **Corrélation Modérée** : Les méthodes montrent une corrélation modérée ({:.3f}), suggérant une certaine cohérence mais aussi des perspectives complémentaires".format(correlation))
            else:
                insights.append("• **Faible Corrélation** : Les méthodes montrent une faible corrélation ({:.3f}), indiquant qu'elles capturent différents aspects de similarité".format(correlation))

        for info in insights:
            st.markdown(info)

    # Export des données filtrées
    st.header("💾 Export des Données Filtrées")
    if st.button("Télécharger les Données Filtrées au format CSV"):
        csv_data = df_filtre.to_csv(index=False)
        st.download_button(
            label="Télécharger le CSV",
            data=csv_data,
            file_name="donnees_filtrees_{}_projets.csv".format(len(df_filtre)),
            mime="text/csv"
        )

# -------------------------------------------------------------------
# ONGLET 2 : DONNÉES FILTRABLES
# -------------------------------------------------------------------
with onglet2:
    st.title("📊 Exploration des Données")
    st.markdown("""
    <div class="highlight-box">
    Explorez les données de similarité avec des filtres avancés. 
    Visualisez et exportez les données selon vos critères de recherche.
    </div>
    """, unsafe_allow_html=True)

    # Filtres spécifiques à cet onglet
    col1, col2 = st.columns(2)
    
    with col1:
        plage_tfidf = st.slider(
            "Plage TF-IDF",
            min_value=float(df['tfidf_score'].min()),
            max_value=float(df['tfidf_score'].max()),
            value=(float(df['tfidf_score'].min()), float(df['tfidf_score'].max())),
            step=0.01,
            key="donnees_tfidf_range"
        )
    
    with col2:
        plage_bert = st.slider(
            "Plage BERT",
            min_value=float(df['bert_score'].min()),
            max_value=float(df['bert_score'].max()),
            value=(float(df['bert_score'].min()), float(df['bert_score'].max())),
            step=0.01,
            key="donnees_bert_range"
        )

    # Options de tri
    col1, col2 = st.columns(2)
    with col1:
        trier_par = st.selectbox("Trier par", ['tfidf_score', 'bert_score'], key="donnees_sort")
    with col2:
        ordre_tri = st.radio("Ordre", ["Décroissant", "Croissant"], key="donnees_order")

    # Appliquer les filtres
    df_donnees_filtre = df[
        (df['tfidf_score'] >= plage_tfidf[0]) & 
        (df['tfidf_score'] <= plage_tfidf[1]) &
        (df['bert_score'] >= plage_bert[0]) & 
        (df['bert_score'] <= plage_bert[1])
    ].copy()

    # Trier les données
    df_donnees_filtre = df_donnees_filtre.sort_values(
        by=trier_par, 
        ascending=(ordre_tri == "Croissant")
    )

    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total des lignes", "{:,}".format(len(df_donnees_filtre)))
    with col2:
        st.metric("Score TF-IDF moyen", "{:.4f}".format(df_donnees_filtre['tfidf_score'].mean()))
    with col3:
        st.metric("Score BERT moyen", "{:.4f}".format(df_donnees_filtre['bert_score'].mean()))
    with col4:
        if len(df_donnees_filtre) > 1:
            corr = df_donnees_filtre['tfidf_score'].corr(df_donnees_filtre['bert_score'])
            st.metric("Corrélation", "{:.4f}".format(corr))
        else:
            st.metric("Corrélation", "N/A")

    # Sélection des colonnes à afficher
    colonnes_disponibles = list(df_donnees_filtre.columns)
    colonnes_affichees = st.multiselect(
        "Colonnes à afficher",
        colonnes_disponibles,
        default=['code_projet_anr', 'cordis_id', 'text_anr', 'text_cordis', 'bert_score', 'tfidf_score'],
        key="donnees_columns"
    )

    # Configuration des colonnes
    config_colonnes = {}
    for col in colonnes_affichees:
        if 'score' in col.lower():
            config_colonnes[col] = st.column_config.NumberColumn(
                col.replace('_', ' ').title(),
                format="%.4f"
            )
        else:
            config_colonnes[col] = st.column_config.TextColumn(
                col.replace('_', ' ').title()
            )

    # Affichage du tableau
    st.subheader("📋 Données ({} lignes)".format(len(df_donnees_filtre)))
    st.dataframe(
        df_donnees_filtre[colonnes_affichees],
        column_config=config_colonnes,
        use_container_width=True,
        height=600
    )

    # Options d'export
    st.subheader("💾 Export")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Télécharger CSV", use_container_width=True):
            csv = df_donnees_filtre.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name="donnees_filtrees_{}_lignes.csv".format(len(df_donnees_filtre)),
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Télécharger Excel", use_container_width=True):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_donnees_filtre.to_excel(writer, index=False, sheet_name='Données filtrées')
            excel_data = output.getvalue()
            
            st.download_button(
                label="Télécharger Excel",
                data=excel_data,
                file_name="donnees_filtrees_{}_lignes.xlsx".format(len(df_donnees_filtre)),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📥 Dataset Complet", use_container_width=True):
            csv_complet = df.to_csv(index=False)
            st.download_button(
                label="Télécharger Dataset Complet",
                data=csv_complet,
                file_name="dataset_complet.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------------
# ONGLET 3 : EXTRACTION TRL
# -------------------------------------------------------------------
with onglet3:
    st.title("🎯 Extraction d'Échelle TRL")
    st.markdown("""
    <div class="highlight-box">
    Analysez la maturité technologique de vos projets de recherche en extrayant automatiquement 
    l'échelle TRL (Technology Readiness Level) à partir des résumés de projets.
    </div>
    """, unsafe_allow_html=True)

    # Gestion de la clé API
    api_key = None
    
    try:
        api_key = st.secrets["openai_api_key"]
        st.success("🔑 Clé API OpenAI chargée depuis les secrets")
    except KeyError:
        st.warning("⚠️ Clé API OpenAI non trouvée dans les secrets")
        
        with st.expander("🔧 Configuration manuelle de l'API", expanded=True):
            api_key = st.text_input(
                "Clé API OpenAI",
                type="password",
                help="Entrez votre clé API OpenAI pour activer l'extraction TRL",
                placeholder="sk-...",
                key="trl_api_key"
            )

    if not api_key:
        st.error("❌ Clé API OpenAI requise pour utiliser cette fonctionnalité")
    else:
        # Guide TRL
        with st.expander("📋 Échelle TRL - Guide de référence"):
            st.markdown("""
            **Technology Readiness Level (TRL) - Définitions :**
            
            | TRL | Niveau | Description |
            |-----|--------|-------------|
            | **TRL 1** | Recherche fondamentale | Principes de base observés et rapportés |
            | **TRL 2** | Formulation du concept | Concept technologique formulé |
            | **TRL 3** | Preuve de concept expérimentale | Preuve de concept analytique et expérimentale |
            | **TRL 4** | Validation en laboratoire | Validation des composants en laboratoire |
            | **TRL 5** | Validation dans un environnement pertinent | Validation des composants dans un environnement pertinent |
            | **TRL 6** | Démonstration dans un environnement pertinent | Démonstration du modèle de système/sous-système |
            | **TRL 7** | Démonstration d'un prototype en environnement opérationnel | Démonstration du prototype de système |
            | **TRL 8** | Système complet et qualifié | Système complet et qualifié par des tests |
            | **TRL 9** | Système opérationnel éprouvé | Système éprouvé par des missions opérationnelles |
            """)

        # Analyse de projet unique
        st.subheader("🔍 Analyse de Projet Unique")
        
        methode_saisie = st.radio(
            "Méthode de saisie :",
            ["Saisie texte", "Upload fichier"],
            key="trl_input_method"
        )
        
        texte_projet = ""
        
        if methode_saisie == "Saisie texte":
            texte_projet = st.text_area(
                "Résumé du projet :",
                height=200,
                placeholder="Collez ici le résumé du projet pour l'analyse TRL...",
                key="trl_project_text"
            )
        else:
            fichier_upload = st.file_uploader(
                "Uploadez le résumé du projet",
                type=['txt'],
                key="trl_file_upload"
            )
            if fichier_upload:
                texte_projet = str(fichier_upload.read(), "utf-8")
                st.text_area("Contenu du fichier :", texte_projet, height=150, disabled=True)

        # Analyse TRL
        if st.button("🎯 Extraire le TRL", type="primary", use_container_width=True):
            if not texte_projet.strip():
                st.error("❌ Veuillez fournir un résumé de projet.")
            else:
                with st.spinner("🤖 Analyse TRL en cours..."):
                    resultat_trl = extraire_trl_avec_openai(texte_projet, api_key)
                    
                    if "TRL" in resultat_trl:
                        st.success("✅ Analyse TRL terminée !")
                        
                        # Affichage du résultat avec style
                        style_resultat = """
                        <div style="
                            background-color: #e8f5e8;
                            padding: 2rem;
                            border-radius: 10px;
                            border-left: 5px solid #4CAF50;
                            text-align: center;
                            margin: 1rem 0;
                        ">
                            <h2 style="color: #2E7D32; margin: 0;">🎯 Résultat</h2>
                            <h1 style="color: #1B5E20; margin: 0.5rem 0; font-size: 3rem;">{}</h1>
                        </div>
                        """.format(resultat_trl)
                        
                        st.markdown(style_resultat, unsafe_allow_html=True)
                        
                        # Contexte basé sur le niveau TRL
                        if resultat_trl != "TRL indéterminé":
                            try:
                                num_trl = int(resultat_trl.split()[-1])
                                
                                if num_trl <= 3:
                                    contexte = "🔬 **Recherche fondamentale** - Le projet est dans une phase de recherche théorique ou de preuve de concept."
                                elif num_trl <= 5:
                                    contexte = "🧪 **Développement en laboratoire** - Le projet valide ses composants dans un environnement contrôlé."
                                elif num_trl <= 7:
                                    contexte = "🔧 **Démonstration et prototypage** - Le projet démontre son système dans un environnement pertinent."
                                else:
                                    contexte = "🚀 **Système mature** - Le projet est proche ou a atteint la maturité opérationnelle."
                                
                                contexte_html = """
                                <div class="highlight-box">
                                {}
                                </div>
                                """.format(contexte)
                                
                                st.markdown(contexte_html, unsafe_allow_html=True)
                                
                            except ValueError:
                                pass
                    else:
                        st.error(f"❌ {resultat_trl}")

# -------------------------------------------------------------------
# ONGLET 4 : COMPARAISON DE RÉSUMÉS
# -------------------------------------------------------------------
with onglet4:
    st.title("💬 Comparaison de Résumés de Projets")
    st.markdown("""
    <div class="highlight-box">
    Comparez deux résumés de projets de recherche en utilisant l'analyse IA. 
    <br><br>
    <strong>🔍 Deux modes disponibles :</strong>
    <ul>
        <li><strong>Analyse Détaillée</strong> : Rapport structuré complet avec domaines, objectifs, méthodologie, impact, et recommandations</li>
        <li><strong>Réponse Simple</strong> : Verdict direct "SIMILAIRES" ou "DIFFÉRENTS" pour une évaluation rapide</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Gestion de la clé API
    api_key_comp = None
    
    try:
        api_key_comp = st.secrets["openai_api_key"]
        st.success("🔑 Clé API OpenAI chargée depuis les secrets")
    except KeyError:
        st.warning("⚠️ Clé API OpenAI non trouvée dans les secrets")
        api_key_comp = st.text_input(
            "Clé API OpenAI",
            type="password",
            help="Entrez votre clé API OpenAI",
            placeholder="sk-...",
            key="comp_api_key"
        )

    if not api_key_comp:
        st.error("❌ Clé API OpenAI requise pour utiliser cette fonctionnalité")
    else:
        # Type d'analyse
        type_analyse = st.selectbox(
            "Type d'analyse",
            ["detailed", "simple"],
            format_func=lambda x: {
                "detailed": "📋 Analyse Détaillée (Rapport complet structuré)",
                "simple": "✅ Réponse Simple (SIMILAIRES / DIFFÉRENTS uniquement)"
            }[x],
            key="type_analyse",
            help="Choisissez le niveau de détail souhaité pour la comparaison"
        )

        # Champs de saisie des résumés
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Résumé Projet ANR")
            resume1 = st.text_area(
                "Collez le résumé du projet ANR :",
                height=300,
                placeholder="Collez ici le résumé du projet ANR...",
                key="resume1"
            )
        
        with col2:
            st.subheader("📝 Résumé Projet CORDIS")
            resume2 = st.text_area(
                "Collez le résumé du projet CORDIS :",
                height=300,
                placeholder="Collez ici le résumé du projet CORDIS...",
                key="resume2"
            )

        # Bouton de comparaison
        if st.button("🔍 Comparer les Projets", type="primary", use_container_width=True):
            if not resume1.strip() or not resume2.strip():
                st.error("❌ Veuillez fournir les deux résumés de projets avant la comparaison.")
            else:
                with st.spinner("🤖 Analyse des projets avec IA..."):
                    try:
                        resultat_comparaison = comparer_resumes_avec_openai(
                            resume1, resume2, api_key_comp, type_analyse
                        )
                        
                        st.success("✅ Comparaison terminée !")
                        
                        # Affichage des résultats
                        st.subheader("🎯 Résultats de l'Analyse IA")
                        if type_analyse == "simple":
                            # Affichage spécial pour le mode simple avec style amélioré
                            if "SIMILAIRES" in resultat_comparaison.upper():
                                st.markdown("""
                                <div style="
                                    background-color: #d4edda;
                                    color: #155724;
                                    padding: 2rem;
                                    border-radius: 10px;
                                    border-left: 5px solid #28a745;
                                    text-align: center;
                                    margin: 1rem 0;
                                ">
                                    <h1 style="margin: 0; font-size: 2.5rem;">✅ SIMILAIRES</h1>
                                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">Les deux projets présentent des similarités significatives</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="
                                    background-color: #f8d7da;
                                    color: #721c24;
                                    padding: 2rem;
                                    border-radius: 10px;
                                    border-left: 5px solid #dc3545;
                                    text-align: center;
                                    margin: 1rem 0;
                                ">
                                    <h1 style="margin: 0; font-size: 2.5rem;">❌ DIFFÉRENTS</h1>
                                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">Les deux projets présentent des différences substantielles</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Affichage pour l'analyse détaillée
                            st.markdown(resultat_comparaison)
                        
                        # Option de téléchargement
                        st.download_button(
                            label="📥 Télécharger le Rapport de Comparaison",
                            data=resultat_comparaison,
                            file_name="comparaison_projets_{}.txt".format(int(time.time())),
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error("❌ Erreur lors de la comparaison : {}".format(str(e)))

        # Exemples d'utilisation
        with st.expander("📋 Exemples de Résumés de Projets"):
            st.markdown("""
            **Exemple Projet ANR (Biotechnologie) :**
            ```
            Titre : BIOMAT-3D
            Objectif : Développement de biomatériaux 3D pour la régénération tissulaire
            
            Résumé :
            Ce projet vise à développer des biomatériaux innovants imprimés en 3D pour 
            applications en médecine régénérative. L'approche combine des polymères 
            biocompatibles avec des cellules souches pour créer des scaffolds fonctionnels. 
            Les applications ciblent la réparation osseuse et cartilagineuse. 
            
            Méthodologie : Impression 3D, culture cellulaire, tests biocompatibilité
            Durée : 42 mois
            Budget : 450k€
            ```
            
            **Exemple Projet CORDIS (Biotechnologie) :**
            ```
            Title: REGEN-BONE
            Objective: Advanced biomaterials for bone regeneration
            
            Abstract:
            This project develops next-generation biomaterials for bone tissue engineering 
            using additive manufacturing techniques. The consortium combines expertise in 
            material science, cell biology, and clinical applications to create 
            patient-specific implants.
            
            Approach: 3D bioprinting, stem cell research, clinical validation
            Duration: 48 months
            Funding: €2.1M
            ```
            
            **Note :** Ces deux projets seraient probablement classés comme "SIMILAIRES" car ils traitent 
            tous deux de biomatériaux pour la régénération osseuse avec des approches d'impression 3D.
            """)

        # Section conseils d'utilisation
        with st.expander("💡 Conseils d'Utilisation"):
            st.markdown("""
            **Pour une analyse optimale :**
            
            ✅ **Résumés recommandés :**
            - Texte de 100 à 1000 mots
            - Inclure les objectifs, méthodologie, et applications
            - Mentionner le domaine de recherche principal
            
            ✅ **Quand choisir "Analyse Détaillée" :**
            - Évaluation approfondie pour décisions stratégiques
            - Besoin de justifications et recommandations
            - Analyse de complémentarité entre projets
            
            ✅ **Quand choisir "Réponse Simple" :**
            - Tri rapide d'un grand nombre de projets
            - Détection de doublons potentiels
            - Classification binaire pour workflows automatisés
            
            ⚠️ **Limitations :**
            - L'IA analyse uniquement le texte fourni
            - La qualité dépend de la précision des résumés
            - Pour des domaines très techniques, une expertise humaine reste recommandée
            """)
        
    # FIN ONGLET 4

# -------------------------------------------------------------------
# FIN DU SCRIPT
# -------------------------------------------------------------------
