# iris/iris.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class IrisProcessor:
    """
    Clase para procesar el dataset Iris
    SOLO contiene lógica de procesamiento
    """
    
    def __init__(self):
        self.iris_data = None
        self.df = None
        self.df_original = None
        self.df_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    # ========== CARGA DE DATOS ==========
    def load_data(self):
        """Carga el dataset Iris desde sklearn"""
        try:
            # Cargar dataset
            self.iris_data = load_iris(as_frame=True)
            
            # Crear DataFrame con datos y target
            self.df = self.iris_data.frame.copy()
            self.df_original = self.df.copy()
            
            # Guardar nombres
            self.feature_names = self.iris_data.feature_names
            self.target_names = self.iris_data.target_names.tolist()
            
            # Agregar nombres de clases
            self.df['species'] = self.df['target'].map({
                0: self.target_names[0],
                1: self.target_names[1],
                2: self.target_names[2]
            })
            
            return {
                'success': True,
                'message': 'Dataset Iris cargado exitosamente',
                'rows': self.df.shape[0],
                'columns': self.df.shape[1] - 2,  # Sin contar target y species
                'features': self.feature_names,
                'classes': self.target_names,
                'samples_per_class': self.df['target'].value_counts().to_dict()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error al cargar el dataset: {str(e)}'
            }
    
    def get_original_data(self):
        """Retorna el DataFrame original"""
        return self.df_original.copy() if self.df_original is not None else None
    
    def get_processed_data(self):
        """Retorna el DataFrame procesado (escalado)"""
        return self.df_scaled.copy() if self.df_scaled is not None else None
    
    def get_feature_data(self):
        """Retorna solo las características (sin target)"""
        if self.df is None:
            return None
        return self.df[self.feature_names].copy()
    
    def get_data_with_target(self):
        """Retorna datos con target y species"""
        if self.df is None:
            return None
        return self.df.copy()
    
    # ========== ESTADÍSTICAS ==========
    def get_basic_statistics(self):
        """Retorna estadísticas básicas del dataset original"""
        if self.df is None:
            return None
        
        return {
            'describe': self.df[self.feature_names].describe(),
            'correlation': self.df[self.feature_names].corr(),
            'class_distribution': self.df['target'].value_counts().to_dict()
        }
    
    def get_scaled_statistics(self):
        """Retorna estadísticas del dataset estandarizado"""
        if self.df_scaled is None:
            return None
        
        return {
            'describe': self.df_scaled[self.feature_names].describe(),
            'correlation': self.df_scaled[self.feature_names].corr()
        }
    
    # ========== ESTANDARIZACIÓN ==========
    def apply_standardization(self):
        """Aplica StandardScaler a las características"""
        if self.df is None:
            return None
        
        try:
            # Obtener solo las características numéricas
            X = self.df[self.feature_names].copy()
            
            # Estadísticas antes de escalar
            stats_before = {
                'mean': X.mean().to_dict(),
                'std': X.std().to_dict(),
                'min': X.min().to_dict(),
                'max': X.max().to_dict()
            }
            
            # Aplicar estandarización
            X_scaled = self.scaler.fit_transform(X)
            
            # Crear DataFrame escalado
            self.df_scaled = pd.DataFrame(
                X_scaled,
                columns=self.feature_names
            )
            
            # Agregar target y species
            self.df_scaled['target'] = self.df['target'].values
            self.df_scaled['species'] = self.df['species'].values
            
            # Estadísticas después de escalar
            stats_after = {
                'mean': self.df_scaled[self.feature_names].mean().to_dict(),
                'std': self.df_scaled[self.feature_names].std().to_dict(),
                'min': self.df_scaled[self.feature_names].min().to_dict(),
                'max': self.df_scaled[self.feature_names].max().to_dict()
            }
            
            return {
                'success': True,
                'message': 'Estandarización aplicada exitosamente',
                'features_scaled': self.feature_names,
                'stats_before': stats_before,
                'stats_after': stats_after,
                'scaler_params': {
                    'mean': self.scaler.mean_.tolist(),
                    'scale': self.scaler.scale_.tolist()
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error en estandarización: {str(e)}'
            }
    
    # ========== DIVISIÓN TRAIN/TEST ==========
    def split_data(self, test_size=0.3, random_state=42):
        """Divide datos en 70% entrenamiento y 30% prueba"""
        if self.df_scaled is None:
            return {
                'success': False,
                'message': 'Primero debe aplicar estandarización'
            }
        
        try:
            # Características y target
            X = self.df_scaled[self.feature_names]
            y = self.df_scaled['target']
            
            # Dividir
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            return {
                'success': True,
                'message': 'Datos divididos exitosamente',
                'X_train_shape': self.X_train.shape,
                'X_test_shape': self.X_test.shape,
                'y_train_shape': self.y_train.shape,
                'y_test_shape': self.y_test.shape,
                'train_percentage': (1 - test_size) * 100,
                'test_percentage': test_size * 100,
                'train_samples': self.X_train.shape[0],
                'test_samples': self.X_test.shape[0],
                'train_class_distribution': self.y_train.value_counts().to_dict(),
                'test_class_distribution': self.y_test.value_counts().to_dict()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error en división de datos: {str(e)}'
            }
    
    # ========== DATOS PARA GRÁFICOS ==========
    def get_scatter_data(self, feature_x='sepal length (cm)', feature_y='petal length (cm)'):
        """Retorna datos para gráfico de dispersión"""
        if self.df is None:
            return None
        
        return {
            'x': self.df[feature_x].values,
            'y': self.df[feature_y].values,
            'target': self.df['target'].values,
            'species': self.df['species'].values,
            'feature_x_name': feature_x,
            'feature_y_name': feature_y
        }
    
    def get_pairplot_data(self):
        """Retorna datos para pairplot"""
        if self.df is None:
            return None
        
        return self.df.copy()
    
    def get_pca_data(self, n_components=3):
        """Aplica PCA y retorna datos para visualización"""
        if self.df is None:
            return None
        
        try:
            # Obtener características
            X = self.df[self.feature_names].values
            
            # Aplicar PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            return {
                'success': True,
                'X_pca': X_pca,
                'target': self.df['target'].values,
                'species': self.df['species'].values,
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'total_variance': sum(pca.explained_variance_ratio_),
                'n_components': n_components
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error en PCA: {str(e)}'
            }
    
    def get_feature_distributions(self):
        """Retorna datos para histogramas de distribución por característica"""
        if self.df is None:
            return None
        
        distributions = {}
        for feature in self.feature_names:
            distributions[feature] = {
                'values': self.df[feature].values,
                'target': self.df['target'].values,
                'species': self.df['species'].values
            }
        
        return distributions
    
    # ========== PROCESO COMPLETO ==========
    def process_all(self):
        """Ejecuta todo el pipeline de procesamiento"""
        if self.df is None:
            return {
                'success': False,
                'message': 'Primero debe cargar el dataset'
            }
        
        results = {}
        
        # 1. Estadísticas originales
        results['original_stats'] = self.get_basic_statistics()
        
        # 2. Estandarización
        results['standardization'] = self.apply_standardization()
        
        if not results['standardization']['success']:
            return results
        
        # 3. División de datos
        results['split'] = self.split_data(test_size=0.3, random_state=42)
        
        if not results['split']['success']:
            return results
        
        # 4. Estadísticas escaladas
        results['scaled_stats'] = self.get_scaled_statistics()
        
        # 5. PCA
        results['pca'] = self.get_pca_data(n_components=3)
        
        results['success'] = True
        results['message'] = 'Pipeline completado exitosamente'
        
        return results
    
    # ========== UTILIDADES ==========
    def is_loaded(self):
        """Verifica si el dataset está cargado"""
        return self.df is not None
    
    def is_scaled(self):
        """Verifica si se aplicó estandarización"""
        return self.df_scaled is not None
    
    def is_split(self):
        """Verifica si se dividieron los datos"""
        return self.X_train is not None
    
    def get_feature_names(self):
        """Retorna nombres de características"""
        return self.feature_names if self.feature_names is not None else []
    
    def get_target_names(self):
        """Retorna nombres de clases"""
        return self.target_names if self.target_names is not None else []