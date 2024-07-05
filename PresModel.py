import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap



def TrainModel (data, labels, pred_year):
    
    #print(data)
    indexes = {2020: 0, 2016: 1, 2012: 2, 2008: 3, 2004: 4, 2000: 5, 1996: 6, 1992: 7, 1988: 8}
    year = indexes[pred_year]
    X_test = data[year*51 : ((year*51)+51), 0:]
    drop_rows = np.arange(year*51, (year*51)+51, 1)
    X_train = np.delete(data, drop_rows, axis=0)
    #print(f'XTRAIN\n{X_train}')
    y_test = labels[indexes[pred_year]*51 : (indexes[pred_year]*51)+51, 0:1]
    y_train = np.delete(labels, drop_rows, axis=0)

    #smote = SMOTE() #add smote values
    #X_sm, y_sm = smote.fit_resample(X_train, y_train)
    #X_train = np.append(X_train, axis=0)
    #y_train = np.append(y_train, axis=0)

    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', min_impurity_decrease=0.01, max_depth=13).fit(X_train, np.ravel(y_train))
    y_pred = forest.predict(X_test)

    return y_pred





def Pred2024(train_data, train_labels, test_data, test_labels):
    pass
    
   



def CreatePredictedMap(results):

    color_dict = {0:'red', 1:'blue'}
   

    fig = plt.figure()
    ax = fig.add_subplot(111)
    map = Basemap(projection='lcc', llcrnrlat=22, llcrnrlon=-119, urcrnrlat=49, urcrnrlon=-64, lat_1=33, lat_2=45, lon_0=-95)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    map.readshapefile('st99_d00', name='states', drawbounds=True)


    postal_order = ['Alabama', "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", 
                    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
                    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
                    "West Virginia", "Wisconsin", 'Wyoming']
    state_names = []
    for shape_dict in map.states_info:
        state_names.append(shape_dict['NAME'])


    ax = plt.gca()

    for state in range(len(postal_order)):
            state_shape = map.states[state_names.index(postal_order[state])]
            fill_state = Polygon(state_shape, facecolor=color_dict[results[state]], edgecolor='black')
            ax.add_patch(fill_state)

    return fig




def CreateActualMap(pred_year, labels):
    
    #get actual labels from results based on index from year selected
    indexes = {2020: 0, 2016: 1, 2012: 2, 2008: 3, 2004: 4, 2000: 5, 1996: 6, 1992: 7, 1988: 8}
    year = indexes[pred_year]
    color_dict = {0:'red', 1:'blue'}

    results = labels[year*51 : ((year*51)+51), 0:1]
    results = np.delete(results, [1, 8, 11])



    fig = plt.figure()
    ax = fig.add_subplot(111)
    map = Basemap(projection='lcc', llcrnrlat=22, llcrnrlon=-119, urcrnrlat=49, urcrnrlon=-64, lat_1=33, lat_2=45, lon_0=-95)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    map.readshapefile('st99_d00', name='states', drawbounds=True)

    postal_order = ['Alabama', "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", 
                    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
                    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
                    "West Virginia", "Wisconsin", 'Wyoming']
    state_names = []
    for shape_dict in map.states_info:
        state_names.append(shape_dict['NAME'])


    ax = plt.gca()

    for state in range(len(postal_order)):
            state_shape = map.states[state_names.index(postal_order[state])]
            fill_state = Polygon(state_shape, facecolor=color_dict[results[state]], edgecolor='black')
            ax.add_patch(fill_state)

    return fig




def main():

    st.set_page_config(page_title="Presidential Election Model", layout='wide')

    #data setup
    df = pd.read_excel('PredictorsReduced.xlsx')
    df = df.to_numpy()
    df_2024 = pd.read_excel('2024Test.xlsx')
    df_2024 = df_2024.to_numpy()

    labels_arr = np.array(df[0:, 12:13], dtype=int)
    data_arr  = np.array(df[0:, 1:12], dtype=float)
    labels_2024 = 0
    data_2024 = 0
    
    #print(labels_arr)
    #print(data_arr)
    
    postal_order = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", 
                    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
                    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
                    "West Virginia", "Wisconsin", "Wyoming"]

    party_ID = {0: 'R', 1: 'D'}

    st.title("Presidential Election Model")

    st.text('This model attempts to predict the presidential vote of individual states based on publicly available demographic information though the use\n of a Random Forest ensemble classifier. When a year is selected from the sidebar the model will be trained on all data avilable\n except the data from that election year. Model predictions will appear in the form of a state district map along with the actual\n results of the election.')


    indexes = {2020: 0, 2016: 1, 2012: 2, 2008: 3, 2004: 4, 2000: 5, 1996: 6, 1992: 7, 1988: 8}
    election_years = np.arange(2024, 1984, -4, dtype=int) 
    year_select = st.selectbox(label="Select Year", options=election_years, index=1)


    if(year_select):
        year = indexes[year_select]
        if (year_select == 2024): 
            pred_results = Pred2024(data_arr, labels_arr, data_2024, labels_2024)
            display_results = np.delete(pred_results, [1, 8, 11]) #remove alaska, DC, and Hawaii from list, not displayed on map, will cause issues
            
            party_results = []
            for state in range(len(pred_results)):
                party_results.append(party_ID[pred_results[state]])

            df_pred_display = pd.DataFrame(data = zip(postal_order, party_results))

            st.header(f'Predicted Results of {year_select} Election')
            st.pyplot(fig_pred)
            st.table(data=df_pred_table)


        else:
            pred_results = TrainModel(data_arr, labels_arr, year_select)
            act_results = (labels_arr[year*51 : (year*51)+51, 0:1])
            act_results = act_results.flatten()
            display_results = np.delete(pred_results, [1, 8, 11]) 
            fig_pred = CreatePredictedMap(display_results)
            fig_act = CreateActualMap(year_select, labels_arr)
            
            party_results = []
            for state in range(len(pred_results)):
                party_results.append(party_ID[pred_results[state]])

            act_party_results = []
            for state in range(len(act_results)):
                act_party_results.append(party_ID[act_results[state]])

            df_pred_table = pd.DataFrame(data = zip(postal_order, party_results), columns=['State', 'Result'])
            df_act_table = pd.DataFrame(data = zip(postal_order, act_party_results), columns=['State', 'Result'])

            col1, col2 = st.columns(2)
            with col1:
                st.header(f'Predicted Results of {year_select} Election')
                st.pyplot(fig_pred)
                st.table(data=df_pred_table)


            with col2:
                st.header(f'Actual Results of {year_select} Election')
                st.pyplot(fig_act)
                st.table(data=df_act_table)







if __name__ == "__main__":
    main()