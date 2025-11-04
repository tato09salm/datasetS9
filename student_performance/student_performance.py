# student_performance/student_performance.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StudentPerformanceProcessor:
    """
    Clase para procesar el dataset Student Performance
    Contiene SOLO la lógica de procesamiento
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_original = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.processing_results = {}
        
    # ========== CARGA DE DATOS ==========
    def load_data(self):
        """Carga el dataset desde el archivo CSV"""
        try:
            self.df = pd.read_csv(self.filepath)
            self.df_original = self.df.copy()
            return {
                'success': True,
                'message': f"Dataset cargado exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas",
                'rows': self.df.shape[0],
                'columns': self.df.shape[1],
                'null_values': self.df.isnull().sum().sum()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error al cargar el dataset: {str(e)}"
            }
    
    def get_original_data(self):
        """Retorna el dataset original"""
        return self.df_original.copy() if self.df_original is not None else None
    
    def get_processed_data(self):
        """Retorna el dataset procesado"""
        return self.df_processed.copy() if self.df_processed is not None else None
    
    def get_column_info(self):
        """Retorna información detallada de las columnas"""
        if self.df_original is None:
            return None
        
        return pd.DataFrame({
            'Tipo de Dato': self.df_original.dtypes,
            'Valores Únicos': self.df_original.nunique(),
            'Valores Nulos': self.df_original.isnull().sum()
        })
    
    def get_target_distribution(self):
        """Retorna la distribución de la variable objetivo G3"""
        if self.df_original is None:
            return None
        
        return self.df_original['G3'].values
    
    # ========== ANÁLISIS CATEGÓRICO ==========
    def analyze_categorical_variables(self):
        """Analiza las variables categóricas del dataset"""
        if self.df is None:
            return None
        
        categorical_cols = []
        categorical_info = {}
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 10:
                if col not in ['G1', 'G2', 'G3']:
                    categorical_cols.append(col)
                    categorical_info[col] = {
                        'unique_values': self.df[col].nunique(),
                        'values': self.df[col].unique().tolist(),
                        'value_counts': self.df[col].value_counts().to_dict()
                    }
        
        return {
            'total': len(categorical_cols),
            'columns': categorical_cols,
            'info': categorical_info
        }
    
    # ========== LIMPIEZA DE DATOS ==========
    def clean_data(self):
        """Elimina duplicados y valores inconsistentes"""
        if self.df is None:
            return None
        
        initial_rows = len(self.df)
        
        # Eliminar duplicados
        duplicates_before = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        duplicates_removed = duplicates_before
        
        # Eliminar filas con valores nulos en G3
        rows_with_null_target = self.df['G3'].isnull().sum()
        self.df = self.df.dropna(subset=['G3'])
        
        # Eliminar filas donde G3 es 0
        rows_with_zero_g3 = (self.df['G3'] == 0).sum()
        self.df = self.df[self.df['G3'] > 0]
        
        # Validar que las notas estén en rango válido (0-20)
        invalid_grades = 0
        for grade_col in ['G1', 'G2', 'G3']:
            invalid = ((self.df[grade_col] < 0) | (self.df[grade_col] > 20)).sum()
            invalid_grades += invalid
            self.df = self.df[(self.df[grade_col] >= 0) & (self.df[grade_col] <= 20)]
        
        final_rows = len(self.df)
        
        results = {
            'duplicates_removed': duplicates_removed,
            'null_target_removed': rows_with_null_target,
            'zero_g3_removed': rows_with_zero_g3,
            'invalid_grades_removed': invalid_grades,
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'total_removed': initial_rows - final_rows
        }
        
        self.processing_results['cleaning'] = results
        return results
    
    # ========== ONE HOT ENCODING ==========
    def apply_encoding(self):
        """Aplica One Hot Encoding a las variables categóricas"""
        if self.df is None:
            return None
        
        # Identificar columnas categóricas
        categorical_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        
        columns_before = list(self.df.columns)
        initial_shape = self.df.shape
        
        # Aplicar One Hot Encoding
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        
        columns_after = list(self.df.columns)
        new_columns = [col for col in columns_after if col not in columns_before]
        
        results = {
            'categorical_cols_encoded': categorical_cols,
            'initial_shape': initial_shape,
            'final_shape': self.df.shape,
            'new_columns': new_columns,
            'columns_added': len(new_columns),
            'total_columns': self.df.shape[1]
        }
        
        self.processing_results['encoding'] = results
        return results
    
    # ========== NORMALIZACIÓN ==========
    def normalize_data(self):
        """Normaliza las variables numéricas"""
        if self.df is None:
            return None
        
        # Variables a normalizar
        numerical_cols = ['age', 'absences', 'G1', 'G2']
        cols_to_normalize = [col for col in numerical_cols if col in self.df.columns]
        
        # Estadísticas antes de normalizar
        stats_before = {}
        for col in cols_to_normalize:
            stats_before[col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        
        # Normalizar
        self.df[cols_to_normalize] = self.scaler.fit_transform(self.df[cols_to_normalize])
        
        # Estadísticas después de normalizar
        stats_after = {}
        for col in cols_to_normalize:
            stats_after[col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        
        results = {
            'normalized_columns': cols_to_normalize,
            'stats_before': stats_before,
            'stats_after': stats_after
        }
        
        self.processing_results['normalization'] = results
        return results
    
    # ========== SEPARACIÓN DE DATOS ==========
    def separate_features_target(self):
        """Separa características (X) y variable objetivo (y)"""
        if self.df is None:
            return None
        
        y = self.df['G3']
        X = self.df.drop('G3', axis=1)
        
        results = {
            'X_shape': X.shape,
            'y_shape': y.shape,
            'feature_columns': list(X.columns),
            'num_features': X.shape[1]
        }
        
        self.processing_results['separation'] = results
        return results
    
    # ========== DIVISIÓN TRAIN/TEST ==========
    def split_train_test(self, test_size=0.2, random_state=42):
        """Divide los datos en entrenamiento (80%) y prueba (20%)"""
        if self.df is None:
            return None
        
        y = self.df['G3']
        X = self.df.drop('G3', axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = {
            'X_train_shape': self.X_train.shape,
            'X_test_shape': self.X_test.shape,
            'y_train_shape': self.y_train.shape,
            'y_test_shape': self.y_test.shape,
            'train_percentage': (1 - test_size) * 100,
            'test_percentage': test_size * 100,
            'train_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0]
        }
        
        self.processing_results['split'] = results
        return results
    
    # ========== ANÁLISIS DE CORRELACIÓN ==========
    def calculate_correlation(self):
        """Calcula la correlación entre G1, G2 y G3"""
        if self.df_original is None:
            return None
        
        grades = self.df_original[['G1', 'G2', 'G3']].copy()
        correlation_matrix = grades.corr()
        
        results = {
            'correlation_matrix': correlation_matrix,
            'G1_G2': float(correlation_matrix.loc['G1', 'G2']),
            'G1_G3': float(correlation_matrix.loc['G1', 'G3']),
            'G2_G3': float(correlation_matrix.loc['G2', 'G3']),
            'data_for_plots': {
                'G1': grades['G1'].values,
                'G2': grades['G2'].values,
                'G3': grades['G3'].values
            }
        }
        
        # Determinar la correlación más fuerte
        correlations = {
            'G1-G2': results['G1_G2'],
            'G1-G3': results['G1_G3'],
            'G2-G3': results['G2_G3']
        }
        strongest = max(correlations, key=correlations.get)
        results['strongest_correlation'] = {
            'pair': strongest,
            'value': correlations[strongest]
        }
        
        self.processing_results['correlation'] = results
        return results
    
    # ========== PROCESO COMPLETO ==========
    def process_all(self):
        """Ejecuta todo el pipeline de procesamiento"""
        if self.df is None:
            return None
        
        results = {}
        
        # 1. Análisis categórico (sin modificar datos)
        results['categorical_analysis'] = self.analyze_categorical_variables()
        
        # 2. Limpieza de datos
        results['cleaning'] = self.clean_data()
        
        # 3. One Hot Encoding
        results['encoding'] = self.apply_encoding()
        
        # 4. Normalización
        results['normalization'] = self.normalize_data()
        
        # 5. Separar X y y
        results['separation'] = self.separate_features_target()
        
        # 6. Dividir datos
        results['split'] = self.split_train_test(test_size=0.2, random_state=42)
        
        # 7. Análisis de correlación (del dataset original)
        results['correlation'] = self.calculate_correlation()
        
        # Guardar dataset procesado
        self.df_processed = self.df.copy()
        
        return results
    
    # ========== UTILIDADES ==========
    def get_statistics(self):
        """Retorna estadísticas del dataset procesado"""
        if self.df_processed is None:
            return None
        
        return self.df_processed.describe()
    
    def is_loaded(self):
        """Verifica si el dataset está cargado"""
        return self.df is not None
    
    def is_processed(self):
        """Verifica si el dataset ha sido procesado"""
        return self.df_processed is not None