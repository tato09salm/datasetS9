"""
Ejercicio 1: Análisis del Dataset “Titanic”
Archivo sugerido: titanic.csv (disponible en Kaggle o Seaborn)
Objetivo: Preparar los datos para un modelo que prediga la supervivencia de los
pasajeros.
Instrucciones:
1. Cargue el dataset con pandas.
2. Elimine columnas irrelevantes como Name, Ticket o Cabin.
3. Verifique valores nulos y reemplácelos con la media o moda según corresponda.
4. Codifique las variables Sex y Embarked.
5. Estandarice las variables numéricas (Age, Fare).
6. Divida los datos en entrenamiento (70%) y prueba (30%) y muestre las
dimensiones resultantes.
Salida esperada:
• Tabla con los primeros 5 registros procesados.
• Impresión de shape de entrenamiento y prueba.
✅                                      ❌ 

"""
#titanic.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TitanicProcessor:
    """
    Clase para procesar el dataset de Titanic
    """
    
    def __init__(self, filepath):
        """Inicializa cargando el dataset"""
        self.filepath = filepath
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    #1: Cargar el dataset 
    def load_data(self):
        """Carga el dataset desde el archivo CSV"""
        try:
            self.df = pd.read_csv(self.filepath)
            return True, "Dataset cargado exitosamente"
        except Exception as e:
            return False, f"Error al cargar el dataset: {str(e)}"
    
    #2: Eliminar columnas irrelevantes 
    def drop_irrelevant_columns(self):
        """Elimina columnas irrelevantes: Name, Ticket, Cabin"""
        columns_to_drop = ['Name', 'Ticket', 'Cabin']
        # Solo elimina las que existan
        columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        
        self.df_processed = self.df.drop(columns=columns_to_drop)
        return self.df_processed.columns.tolist()
    
    #3: Verificar y rellenar valores nulos
    def handle_missing_values(self):
        """
        Verifica valores nulos y los rellena:
        - Columnas numéricas: con la media
        - Columnas categóricas: con la moda
        """
        missing_before = self.df_processed.isnull().sum()
        
        # Rellenar valores nulos en columnas numéricas con la media
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df_processed[col].isnull().sum() > 0:
                self.df_processed[col].fillna(self.df_processed[col].mean(), inplace=True)
        
        # Rellenar valores nulos en columnas categóricas con la moda
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df_processed[col].isnull().sum() > 0:
                self.df_processed[col].fillna(self.df_processed[col].mode()[0], inplace=True)
        
        missing_after = self.df_processed.isnull().sum()
        
        return missing_before, missing_after
    
    #4: Codificar variables categóricas
    def encode_categorical(self):
        """
        Codifica las variables Sex y Embarked usando LabelEncoder
        """
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        
        encoding_info = {}
        
        # Codificar Sex
        #0 a "female" y 1 a "male"
        if 'Sex' in self.df_processed.columns:
            self.df_processed['Sex'] = le_sex.fit_transform(self.df_processed['Sex'])
            encoding_info['Sex'] = dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))
        
        # Codificar Embarked
        #0 a "C", 1 a "Q", y 2 a "S".
        if 'Embarked' in self.df_processed.columns:
            self.df_processed['Embarked'] = le_embarked.fit_transform(self.df_processed['Embarked'])
            encoding_info['Embarked'] = dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))
        
        return encoding_info
    
    #5: Estandarizar variables numéricas
    def standardize_features(self):
        """
        Estandariza las variables numéricas Age y Fare usando StandardScaler
        """
        features_to_scale = ['Age', 'Fare']
        # Solo escala las que existan
        features_to_scale = [col for col in features_to_scale if col in self.df_processed.columns]
        
        if features_to_scale:
            self.df_processed[features_to_scale] = self.scaler.fit_transform(
                self.df_processed[features_to_scale]
            )
        
        return features_to_scale
    
    #6: Dividir en entrenamiento y prueba
    def split_data(self, test_size=0.3, random_state=42):
        """
        Divide los datos en conjunto de entrenamiento (70%) y prueba (30%)
        """
        # Separar características (X) y variable objetivo (y)
        if 'Survived' in self.df_processed.columns:
            X = self.df_processed.drop('Survived', axis=1)
            y = self.df_processed['Survived']
        else:
            raise ValueError("La columna 'Survived' no existe en el dataset")
        
        # Dividir los datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return {
            'X_train_shape': self.X_train.shape,
            'X_test_shape': self.X_test.shape,
            'y_train_shape': self.y_train.shape,
            'y_test_shape': self.y_test.shape
        }
    
    # ===== MÉTODO AUXILIAR: Ejecutar todo el pipeline =====
    def process_all(self):
        """
        Ejecuta todo el pipeline de procesamiento
        """
        results = {}
        
        # 1. Cargar datos
        success, message = self.load_data()
        results['load'] = {'success': success, 'message': message}
        if not success:
            return results
        
        # 2. Eliminar columnas
        results['dropped_columns'] = self.drop_irrelevant_columns()
        
        # 3. Manejar valores nulos
        missing_before, missing_after = self.handle_missing_values()
        results['missing_values'] = {
            'before': missing_before.to_dict(),
            'after': missing_after.to_dict()
        }
        
        # 4. Codificar categóricas
        results['encoding'] = self.encode_categorical()
        
        # 5. Estandarizar
        results['scaled_features'] = self.standardize_features()
        
        # 6. Dividir datos
        results['split_info'] = self.split_data()
        
        return results
    
    # ===== MÉTODOS PARA OBTENER RESULTADOS =====
    def get_original_data(self):
        """Retorna el dataset original"""
        return self.df
    
    def get_processed_data(self):
        """Retorna el dataset procesado"""
        return self.df_processed
    
    def get_train_test_data(self):
        """Retorna los conjuntos de entrenamiento y prueba"""
        return self.X_train, self.X_test, self.y_train, self.y_test