from sklearn.model_selection import train_test_split

def load_titanic_dataset(data): 
    ''' Sex to number -> male : 1, female : 0 '''
    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    
    ''' Fill missing data and normalize age'''
    data['Age'] = data['Age'].fillna(100)
    data['Age'] = data['Age'].apply(lambda a: a / 100)
    
    ''' Normalize pclass'''
    data['Pclass'] = data['Pclass'].apply(lambda pc: pc / 3)
    
    ''' Parch -> yes or no'''
    data['Parch'] = data['Parch'].apply(lambda pa: 1 if pa > 0 else 0 )
    
    ''' Normalize fare'''
    data['Fare'] = data['Fare'].apply(lambda f: f / 500)
    
    ''' Embarked to number'''
    data['Embarked'] = data['Embarked'].apply(lambda e: 0 if e == 'S' else 0.5 if e == 'C' else 0)
    
    dataset_x = data[['Sex', 'Age', 'Pclass', 'Parch', 'Fare', 'Embarked']]
    dataset_x = dataset_x.as_matrix()
    
    if 'Survived' in data.keys():
        '''Add "Deceased" column to classification'''
        data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
        
        dataset_t = data[['Deceased', 'Survived']]
        dataset_t = dataset_t.as_matrix()
    else:
        dataset_t = data['PassengerId']
        dataset_t = dataset_t.as_matrix()
    
    return dataset_x, dataset_t