import timeit
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

# Função para ler os dados
@st.cache_resource(show_spinner= True)
def load_data(file_data):

  return pd.read_csv(file_data, infer_datetime_format=True, parse_dates=['DiaCompra'])


# Função para transformar os dados
@st.cache_resource(show_spinner= True)
def transformar_dados(df):

  df_recencia = df.groupby(by='ID_cliente',
                                  as_index=False)['DiaCompra'].max()
  df_recencia.columns = ['ID_cliente', 'DiaUltimaCompra']

  dia_atual = df['DiaCompra'].max()

  df_recencia['Recencia'] = df_recencia['DiaUltimaCompra'].apply(
      lambda x: (dia_atual - x).days)

  df_recencia.drop('DiaUltimaCompra', axis=1, inplace=True)

  df_frequencia = df[['ID_cliente', 'CodigoCompra'
                              ]].groupby('ID_cliente').count().reset_index()
  df_frequencia.columns = ['ID_cliente', 'Frequencia']

  df_valor = df[['ID_cliente', 'ValorTotal'
                        ]].groupby('ID_cliente').sum().reset_index()
  df_valor.columns = ['ID_cliente', 'Valor']

  df_RF = df_recencia.merge(df_frequencia, on='ID_cliente')

  df_RFV = df_RF.merge(df_valor, on='ID_cliente')
  df_RFV.set_index('ID_cliente', inplace=True)

  return df_RFV.reset_index()

# Função de clusterização
@st.cache_resource(show_spinner= True)
def clusterizar(df, n):

  cols = ['Recencia', 'Frequencia', 'Valor']

  scaler = MinMaxScaler(feature_range=(0,1))

  scaled_data = scaler.fit_transform(df[cols])

  clustering = AgglomerativeClustering(n_clusters=n).fit(scaled_data)

  df['Grupo'] = clustering.labels_

  df_cluster = df.copy()

  return df_cluster

# Função para converter o df para csv
@st.cache_data
def convert_df(df):
   return df.to_csv(index=True).encode('utf-8')

# _____________________________________________________________________________   

def main():
    st.set_page_config(page_title = 'Clusterização dos Clientes', \
        page_icon = '/content/telmarketing_icon.png',
        layout ='wide',
        initial_sidebar_state='expanded')
    st.title('Clusterização dos Clientes')
    st.markdown('---')

    st.sidebar.image('https://marketingpordados.com/wp-content/uploads/2021/08/Clusterizacao-2.jpg')

    data_file_1 = st.sidebar.file_uploader("Dados dos Clientes",
                                            type=['csv','xlsx'])
    st.sidebar.subheader('Defina a quantidade de grupos')                                        

    with st.sidebar.form(key='my_form'):

      # SELECIONAR A QUANTIDADE DE CLUSTERS
      
      n = st.select_slider('Quantidade de grupos', [2, 3, 4, 5, 6], value=2)

      st.form_submit_button(label='Aplicar')                                        

    if (data_file_1 is not None):

      start = timeit.default_timer()
      bank_raw = load_data(data_file_1)
        
      st.write('Time: ', timeit.default_timer() - start)  
      bank = bank_raw.copy()
      st.header('I - Tratamento dos Dados')
      st.subheader('1 - Dados Brutos')
      st.dataframe(bank_raw)

    try:      

      dataset = transformar_dados(bank_raw)  
      st.subheader('2 - RFV dos Clientes')      
      st.dataframe(dataset)
      bank_csv = convert_df(dataset)
      st.download_button(label='📥 Download',
                        data=bank_csv,
                        file_name= 'dataset.csv')

      st.markdown("---")
      st.header('II - Clusterização dos Dados')
      st.subheader('1 - Relação dos Clientes Agrupados')
      dataset_2 = dataset.copy()  
      dataset_cluster = clusterizar(dataset_2, n)
      st.dataframe(dataset_cluster)
      bank_csv_2 = convert_df(dataset_cluster)
      st.download_button(label='📥 Download',
                        data=bank_csv_2,
                        file_name= 'dataset_cluster.csv')

      st.markdown("---")
      st.subheader('2 - Quantidade dos Grupos')
      cols_2 = dataset_cluster.columns[1:]
      dataset_group = pd.DataFrame(dataset_cluster['Grupo'].value_counts())
      st.dataframe(dataset_group)

      st.subheader('Quantidade')
      st.bar_chart(dataset_group)


      
      st.subheader('3 - RFV Médio por Grupo')
      cols_2 = dataset_cluster.columns[1:]
      dataset_group_2 = pd.DataFrame(dataset_cluster[cols_2].groupby('Grupo').mean())
      st.dataframe(dataset_group_2)

      colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

      col1, col2, col3 = st.columns(3)

      with col1:

        st.markdown('#### Recência Média')
        st.bar_chart(dataset_group_2['Recencia'], color=colors[0])

      with col2:

        st.markdown('#### Frequência Média')
        st.bar_chart(dataset_group_2['Frequencia'], color=colors[1])

      with col3:

        st.markdown('#### Valor Médio')
        st.bar_chart(dataset_group_2['Valor'], color=colors[2])



    except:

      st.write('## Carregue os dados')

    #  PLOTS  
      


if __name__ == '__main__':
	main()