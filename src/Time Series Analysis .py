import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import IsolationForest


class TimeSeriesGitHub:
def __init__(self, repo_path, github_username, github_email):
    """
    Initialize the project with GitHub credentials
    
    Args:
        repo_path (str): Path to local repository
        github_username (str): GitHub username
        github_email (str): GitHub email
    """
    self.repo_path = Path(repo_path).absolute()  # Get absolute path
    self.username = github_username
    self.email = github_email
    
    # Create directory if it doesn't exist
    os.makedirs(self.repo_path, exist_ok=True)
    
    # Initialize git repository
    try:
        self.repo = git.Repo(self.repo_path)
    except git.exc.InvalidGitRepositoryError:
        self.repo = git.Repo.init(self.repo_path)
        # Configure git user
        with self.repo.config_writer() as git_config:
            git_config.set_value('user', 'name', self.username)
            git_config.set_value('user', 'email', self.email)
    
    # Setup logger after directory is created
    self.logger = self._setup_logger()
    
    # Initialize models
    self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    self.prophet_model = Prophet(interval_width=0.95)
    
    def _setup_logger(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.repo_path / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def generate_sample_data(self, start_date='2024-01-01', n_points=365):
        """
        Generate sample time series data with anomalies
        
        Args:
            start_date (str): Start date for the time series
            n_points (int): Number of data points to generate
            
        Returns:
            pd.DataFrame: Sample time series data
        """
        dates = pd.date_range(start=start_date, periods=n_points, freq='D')
        t = np.linspace(0, 4*np.pi, n_points)
        
        # Generate base signal with trend and seasonality
        trend = np.linspace(0, 10, n_points)
        seasonality = 5 * np.sin(t) + 2 * np.cos(2*t)
        noise = np.random.normal(0, 0.5, n_points)
        
        signal = trend + seasonality + noise
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_points, size=5, replace=False)
        signal[anomaly_indices] += np.random.uniform(5, 10, size=5)
        
        return pd.DataFrame({
            'ds': dates,
            'y': signal
        })
    
    def detect_anomalies(self, data):
        """
        Detect anomalies using Isolation Forest
        
        Args:
            data (pd.DataFrame): Time series data with 'ds' and 'y' columns
            
        Returns:
            pd.DataFrame: Data with anomaly labels
        """
        X = np.array(data['y']).reshape(-1, 1)
        anomalies = self.isolation_forest.fit_predict(X)
        
        results = data.copy()
        results['is_anomaly'] = anomalies == -1
        
        return results
    
    def forecast(self, data, periods=30):
        """
        Generate forecasts using Prophet
        
        Args:
            data (pd.DataFrame): Data with 'ds' and 'y' columns
            periods (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: Forecast results
        """
        self.prophet_model.fit(data)
        future_dates = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future_dates)
        
        return forecast
    
    def plot_results(self, data, forecast):
        """
        Create visualizations of the analysis
        
        Args:
            data (pd.DataFrame): Original data with anomalies
            forecast (pd.DataFrame): Prophet forecast results
        """
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # Plot original data and anomalies
        plt.subplot(2, 1, 1)
        plt.plot(data['ds'], data['y'], label='Original Data')
        anomalies = data[data['is_anomaly']]
        plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies')
        plt.title('Time Series with Detected Anomalies')
        plt.legend()
        
        # Plot forecast
        plt.subplot(2, 1, 2)
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'], 
                        alpha=0.3, 
                        label='Confidence Interval')
        plt.title('Time Series Forecast')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.repo_path / 'analysis_results.png')
        plt.close()
    
    def save_results(self, data, forecast):
        """
        Save analysis results to files
        
        Args:
            data (pd.DataFrame): Data with anomalies
            forecast (pd.DataFrame): Forecast results
        """
        # Save data with anomalies
        data.to_csv(self.repo_path / 'anomalies.csv', index=False)
        
        # Save forecast
        forecast.to_csv(self.repo_path / 'forecast.csv', index=False)
        
        # Create summary report
        with open(self.repo_path / 'analysis_report.md', 'w') as f:
            f.write('# Time Series Analysis Report\n\n')
            f.write(f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            f.write('## Anomaly Detection\n')
            f.write(f'Total data points: {len(data)}\n')
            f.write(f'Anomalies detected: {data["is_anomaly"].sum()}\n\n')
            
            f.write('## Forecast Summary\n')
            f.write(f'Forecast periods: {len(forecast) - len(data)}\n')
            f.write(f'Average forecast value: {forecast["yhat"].mean():.2f}\n')
            f.write(f'Forecast range: {forecast["yhat"].min():.2f} to {forecast["yhat"].max():.2f}\n')
    
    def commit_and_push(self):
        """Commit changes and push to GitHub"""
        try:
            # Add all files
            self.repo.index.add('*')
            
            # Create commit
            commit_message = f"Update analysis results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.repo.index.commit(
                commit_message,
                author=git.Actor(self.username, self.email)
            )
            
            self.logger.info("Successfully committed changes")
            
            # Push changes if remote exists
            if self.repo.remotes:
                self.repo.remote().push()
                self.logger.info("Successfully pushed to remote repository")
            else:
                self.logger.info("No remote repository configured. Please add one and push manually.")
                
        except Exception as e:
            self.logger.error(f"Error in commit_and_push: {str(e)}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Generate sample data
            self.logger.info("Generating sample data...")
            data = self.generate_sample_data()
            
            # Detect anomalies
            self.logger.info("Detecting anomalies...")
            data_with_anomalies = self.detect_anomalies(data)
            
            # Generate forecast
            self.logger.info("Generating forecast...")
            forecast_results = self.forecast(data)
            
            # Create visualizations
            self.logger.info("Creating visualizations...")
            self.plot_results(data_with_anomalies, forecast_results)
            
            # Save results
            self.logger.info("Saving results...")
            self.save_results(data_with_anomalies, forecast_results)
            
            # Commit and push changes
            self.logger.info("Committing and pushing changes...")
            self.commit_and_push()
            
            self.logger.info("Analysis completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Configuration
    REPO_PATH = "time_series_analysis" 
    GITHUB_USERNAME = "Tondwani"
    GITHUB_EMAIL = "craigmangaladzi@gmail.com"
    GITHUB_REPO_URL = "https://github.com/Tondwani/time_series_analysis.git"
    
    # Initialize and run analysis
    analyzer = TimeSeriesGitHub(
        repo_path=REPO_PATH,
        github_username=GITHUB_USERNAME,
        github_email=GITHUB_EMAIL
    )
    
    try:
        if 'origin' not in [remote.name for remote in analyzer.repo.remotes]:
            analyzer.repo.create_remote('origin', GITHUB_REPO_URL)
        
        # Run analysis
        analyzer.run_analysis()
        
    except Exception as e:
        print(f"Error setting up repository: {str(e)}")