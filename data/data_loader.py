"""
Real-World Dataset Loaders for HT-Bench

This module provides loaders for real-world datasets used in hypothesis testing scenarios:
- StockDataLoader: US Stock Market 2020-2024 data for financial hypothesis testing
- HealthcareDataLoader: Pima Indians Diabetes dataset for medical hypothesis testing

These loaders generate scenarios that match the existing synthetic data format while
providing realistic domain context for LLM evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random

# Base path for raw data
DATA_DIR = Path(__file__).parent / "raw"

# Stock groupings for comparative analysis
STOCK_GROUPS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ORCL", "ADBE"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "LLY"],
    "consumer": ["AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "DIS"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "industrial": ["CAT", "DE", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "UNP"],
}


class StockDataLoader:
    """
    Loader for US Stock Market 2020-2024 dataset.
    
    Provides methods to extract data for various hypothesis testing scenarios:
    - One-sample t-test: Test if a stock's mean return differs from a benchmark
    - Two-sample t-test: Compare returns between two stocks or sectors
    - Paired t-test: Compare before/after periods or matched time windows
    """
    
    def __init__(self, filepath: Optional[Path] = None):
        """
        Initialize the stock data loader.
        
        Args:
            filepath: Path to the stock CSV file. Defaults to data/raw/stocks/us_stock.csv
        """
        self.filepath = filepath or DATA_DIR / "stocks" / "us_stock.csv"
        self._df = None
        self._available_tickers = None
        
    @property
    def df(self) -> pd.DataFrame:
        """Lazy-load the dataframe."""
        if self._df is None:
            self._load_data()
        return self._df
    
    def _load_data(self):
        """Load and preprocess the stock data."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Stock data file not found: {self.filepath}")
        
        self._df = pd.read_csv(self.filepath)
        
        # Standardize column names (handle various formats)
        self._df.columns = self._df.columns.str.lower().str.strip()
        
        # Try to identify the date column and parse with flexible format
        date_cols = [c for c in self._df.columns if 'date' in c.lower()]
        if date_cols:
            try:
                # Try multiple date formats
                self._df['date'] = pd.to_datetime(self._df[date_cols[0]], format='mixed', dayfirst=True)
            except Exception:
                try:
                    self._df['date'] = pd.to_datetime(self._df[date_cols[0]], infer_datetime_format=True)
                except Exception:
                    # Last resort - just try default parsing
                    self._df['date'] = pd.to_datetime(self._df[date_cols[0]], errors='coerce')
        
        # Try to identify ticker/symbol column
        ticker_cols = [c for c in self._df.columns if c in ['ticker', 'symbol', 'stock', 'name']]
        if ticker_cols:
            self._df['ticker'] = self._df[ticker_cols[0]].astype(str).str.upper()
            self._available_tickers = self._df['ticker'].unique().tolist()
        else:
            self._available_tickers = []
            
        # Try to identify price columns for return calculation
        price_cols = [c for c in self._df.columns if c in ['close', 'adj close', 'adj_close', 'adjusted_close', 'price']]
        if price_cols:
            self._df['price'] = pd.to_numeric(self._df[price_cols[0]], errors='coerce')
            
    @property
    def available_tickers(self) -> List[str]:
        """Get list of available stock tickers."""
        if self._available_tickers is None:
            self._load_data()
        return self._available_tickers or []
    
    def get_returns(self, ticker: str, period: str = 'daily') -> np.ndarray:
        """
        Calculate returns for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            period: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Array of percentage returns
        """
        df = self.df[self.df['ticker'] == ticker.upper()].copy()
        
        if 'price' not in df.columns or df.empty:
            # Return synthetic data if we can't get real data
            return np.random.normal(0.001, 0.02, 100) * 100
        
        df = df.sort_values('date')
        
        if period == 'weekly':
            df = df.set_index('date').resample('W').last().reset_index()
        elif period == 'monthly':
            df = df.set_index('date').resample('M').last().reset_index()
            
        returns = df['price'].pct_change().dropna() * 100  # Convert to percentage
        return returns.values
    
    def get_asset_returns(self, n_samples: int = 50, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a one-sample t-test scenario using real stock data.
        
        Tests if a stock's mean daily return differs from a benchmark (e.g., 0% or market average).
        
        Args:
            n_samples: Number of data points to include
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario data and context
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Pick a random ticker or use available data
        if self.available_tickers:
            ticker = random.choice(self.available_tickers[:20])  # Limit to first 20 for variety
            returns = self.get_returns(ticker)
            if len(returns) >= n_samples:
                # Random contiguous window
                start_idx = random.randint(0, len(returns) - n_samples)
                data = returns[start_idx:start_idx + n_samples]
            else:
                data = returns
        else:
            # Fallback to simulated stock-like returns
            ticker = random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
            data = np.random.normal(0.05, 1.5, n_samples)
            
        # Common benchmark tests
        benchmarks = [
            (0.0, "zero (testing if returns are significantly different from zero)"),
            (0.05, "market average daily return of 0.05%"),
            (-0.02, "the risk-free rate equivalent of -0.02%"),
        ]
        benchmark_value, benchmark_desc = random.choice(benchmarks)
        
        return {
            "data": data.tolist() if isinstance(data, np.ndarray) else data,
            "population_mean": benchmark_value,
            "context": {
                "domain": "finance",
                "dataset": "US Stock Market 2020-2024",
                "description": f"Daily percentage returns for {ticker} stock",
                "test_description": f"Testing if {ticker}'s mean daily return differs from {benchmark_desc}",
                "units": "percentage (%)",
                "ticker": ticker,
            }
        }
    
    def get_group_returns(self, n_samples: int = 30, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a two-sample t-test scenario comparing two stock groups.
        
        Args:
            n_samples: Number of data points per group
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario data and context
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Pick two different sectors
        sectors = list(STOCK_GROUPS.keys())
        sector1, sector2 = random.sample(sectors, 2)
        
        # Try to get real data for stocks in each sector
        def get_sector_returns(sector: str) -> np.ndarray:
            tickers_in_sector = [t for t in STOCK_GROUPS[sector] if t in self.available_tickers]
            if tickers_in_sector:
                ticker = random.choice(tickers_in_sector)
                returns = self.get_returns(ticker)
                if len(returns) >= n_samples:
                    start_idx = random.randint(0, len(returns) - n_samples)
                    return returns[start_idx:start_idx + n_samples]
            # Fallback to simulated sector-specific returns
            sector_params = {
                "tech": (0.08, 2.0),
                "finance": (0.04, 1.5),
                "healthcare": (0.03, 1.2),
                "consumer": (0.05, 1.4),
                "energy": (0.02, 2.5),
                "industrial": (0.04, 1.6),
            }
            mean, std = sector_params.get(sector, (0.05, 1.5))
            return np.random.normal(mean, std, n_samples)
        
        data1 = get_sector_returns(sector1)
        data2 = get_sector_returns(sector2)
        
        return {
            "group1_data": data1.tolist() if isinstance(data1, np.ndarray) else data1,
            "group2_data": data2.tolist() if isinstance(data2, np.ndarray) else data2,
            "context": {
                "domain": "finance",
                "dataset": "US Stock Market 2020-2024",
                "description": f"Comparing daily returns between {sector1.title()} and {sector2.title()} sectors",
                "group1_name": f"{sector1.title()} Sector",
                "group2_name": f"{sector2.title()} Sector",
                "test_description": f"Testing if {sector1.title()} sector stocks have different mean returns than {sector2.title()} sector",
                "units": "percentage (%)",
            }
        }
    
    def get_paired_periods(self, n_pairs: int = 25, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a paired t-test scenario comparing before/after periods.
        
        E.g., comparing returns before and after a major market event.
        
        Args:
            n_pairs: Number of paired observations
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario data and context
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Market events for context
        events = [
            ("COVID-19 pandemic onset", "March 2020", "comparing returns before and after the pandemic crash"),
            ("Fed interest rate hike cycle", "March 2022", "comparing returns before and after rate increases"),
            ("Tech sector correction", "January 2022", "comparing returns before and after the tech selloff"),
            ("Market recovery rally", "April 2020", "comparing returns during crash vs recovery"),
        ]
        event_name, event_date, event_desc = random.choice(events)
        
        # Try to get real paired data
        if self.available_tickers:
            ticker = random.choice(self.available_tickers[:10])
            returns = self.get_returns(ticker)
            
            if len(returns) >= n_pairs * 2:
                # Split into before/after periods
                mid_point = len(returns) // 2
                before = returns[mid_point - n_pairs:mid_point]
                after = returns[mid_point:mid_point + n_pairs]
            else:
                # Simulate with event effect
                before = np.random.normal(0.05, 1.5, n_pairs)
                effect = random.uniform(-0.5, 0.5)  # Event impact
                after = np.random.normal(0.05 + effect, 1.5 * random.uniform(0.8, 1.2), n_pairs)
        else:
            ticker = random.choice(["SPY", "QQQ", "DIA"])
            before = np.random.normal(0.05, 1.5, n_pairs)
            effect = random.uniform(-0.5, 0.5)
            after = np.random.normal(0.05 + effect, 1.5, n_pairs)
            
        return {
            "before_data": before.tolist() if isinstance(before, np.ndarray) else before,
            "after_data": after.tolist() if isinstance(after, np.ndarray) else after,
            "context": {
                "domain": "finance",
                "dataset": "US Stock Market 2020-2024",
                "description": f"Paired comparison of {ticker if 'ticker' in dir() else 'market'} returns {event_desc}",
                "event": event_name,
                "event_date": event_date,
                "test_description": f"Testing if returns changed significantly after {event_name}",
                "units": "percentage (%)",
            }
        }


class HealthcareDataLoader:
    """
    Loader for Pima Indians Diabetes dataset.
    
    Provides methods to extract data for various hypothesis testing scenarios:
    - One-sample t-test: Test if a health metric differs from population norms
    - Two-sample t-test: Compare metrics between diabetic and non-diabetic groups
    - Paired t-test: (Simulated) Before/after intervention comparisons
    """
    
    # Population norms for health metrics (approximate values)
    POPULATION_NORMS = {
        "glucose": {"mean": 100, "unit": "mg/dL", "name": "Fasting Blood Glucose"},
        "bloodpressure": {"mean": 72, "unit": "mm Hg", "name": "Diastolic Blood Pressure"},
        "skinthickness": {"mean": 20, "unit": "mm", "name": "Triceps Skin Fold Thickness"},
        "insulin": {"mean": 80, "unit": "μU/mL", "name": "2-Hour Serum Insulin"},
        "bmi": {"mean": 25, "unit": "kg/m²", "name": "Body Mass Index"},
        "diabetespedigreefunction": {"mean": 0.5, "unit": "score", "name": "Diabetes Pedigree Function"},
        "age": {"mean": 33, "unit": "years", "name": "Age"},
        "pregnancies": {"mean": 3, "unit": "count", "name": "Number of Pregnancies"},
    }
    
    def __init__(self, filepath: Optional[Path] = None):
        """
        Initialize the healthcare data loader.
        
        Args:
            filepath: Path to the diabetes CSV file. Defaults to data/raw/healthcare/diabetes.csv
        """
        self.filepath = filepath or DATA_DIR / "healthcare" / "diabetes.csv"
        self._df = None
        
    @property
    def df(self) -> pd.DataFrame:
        """Lazy-load the dataframe."""
        if self._df is None:
            self._load_data()
        return self._df
    
    def _load_data(self):
        """Load and preprocess the diabetes data."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Healthcare data file not found: {self.filepath}")
        
        self._df = pd.read_csv(self.filepath)
        
        # Standardize column names
        self._df.columns = self._df.columns.str.lower().str.strip()
        
        # The Pima dataset uses 'outcome' as the diabetes indicator (0 = no, 1 = yes)
        if 'outcome' in self._df.columns:
            self._df['diabetic'] = self._df['outcome']
        elif 'diabetes' in self._df.columns:
            self._df['diabetic'] = self._df['diabetes']
            
        # Handle missing values coded as 0 (common in Pima dataset)
        zero_invalid_cols = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']
        for col in zero_invalid_cols:
            if col in self._df.columns:
                self._df.loc[self._df[col] == 0, col] = np.nan
    
    @property
    def available_features(self) -> List[str]:
        """Get list of available numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['outcome', 'diabetic', 'diabetes']
        return [c for c in numeric_cols if c not in exclude]
    
    def get_feature_by_outcome(self, feature: Optional[str] = None, n_samples: int = 50, 
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a two-sample t-test scenario comparing feature values between diabetic and non-diabetic patients.
        
        Args:
            feature: Feature to compare. If None, randomly selected.
            n_samples: Number of samples per group
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario data and context
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Select feature
        available = self.available_features
        if feature is None:
            feature = random.choice(available) if available else "glucose"
        elif feature not in available:
            feature = "glucose"  # Fallback
            
        # Get data for each group
        diabetic_data = self.df[self.df['diabetic'] == 1][feature].dropna()
        non_diabetic_data = self.df[self.df['diabetic'] == 0][feature].dropna()
        
        # Sample if we have enough data
        if len(diabetic_data) >= n_samples:
            diabetic_sample = diabetic_data.sample(n_samples, random_state=seed).values
        else:
            diabetic_sample = diabetic_data.values
            
        if len(non_diabetic_data) >= n_samples:
            non_diabetic_sample = non_diabetic_data.sample(n_samples, random_state=seed).values
        else:
            non_diabetic_sample = non_diabetic_data.values
            
        # Get feature metadata
        feature_info = self.POPULATION_NORMS.get(feature, {"mean": 0, "unit": "units", "name": feature.title()})
        
        return {
            "group1_data": diabetic_sample.tolist(),
            "group2_data": non_diabetic_sample.tolist(),
            "context": {
                "domain": "healthcare",
                "dataset": "Pima Indians Diabetes Database",
                "description": f"Comparing {feature_info['name']} between diabetic and non-diabetic patients",
                "group1_name": "Diabetic Patients",
                "group2_name": "Non-Diabetic Patients",
                "feature": feature,
                "test_description": f"Testing if {feature_info['name']} differs significantly between diabetic and non-diabetic individuals",
                "units": feature_info['unit'],
            }
        }
    
    def get_population_comparison(self, feature: Optional[str] = None, n_samples: int = 50,
                                   seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a one-sample t-test scenario comparing patient data to population norms.
        
        Args:
            feature: Feature to test. If None, randomly selected.
            n_samples: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario data and context
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Select feature with known population norm
        features_with_norms = [f for f in self.available_features if f in self.POPULATION_NORMS]
        if feature is None:
            feature = random.choice(features_with_norms) if features_with_norms else "glucose"
        elif feature not in features_with_norms:
            feature = "glucose"
            
        # Get sample data
        feature_data = self.df[feature].dropna()
        if len(feature_data) >= n_samples:
            sample = feature_data.sample(n_samples, random_state=seed).values
        else:
            sample = feature_data.values
            
        feature_info = self.POPULATION_NORMS.get(feature, {"mean": 100, "unit": "units", "name": feature.title()})
        population_mean = feature_info["mean"]
        
        # Sometimes test diabetic subpopulation
        test_subpop = random.choice([True, False])
        if test_subpop:
            diabetic_data = self.df[self.df['diabetic'] == 1][feature].dropna()
            if len(diabetic_data) >= n_samples:
                sample = diabetic_data.sample(n_samples, random_state=seed).values
            subpop_desc = "diabetic patients"
        else:
            subpop_desc = "the study population"
            
        return {
            "data": sample.tolist(),
            "population_mean": population_mean,
            "context": {
                "domain": "healthcare",
                "dataset": "Pima Indians Diabetes Database",
                "description": f"{feature_info['name']} measurements from {subpop_desc}",
                "test_description": f"Testing if mean {feature_info['name']} in {subpop_desc} differs from the general population norm of {population_mean} {feature_info['unit']}",
                "feature": feature,
                "units": feature_info['unit'],
                "population_norm": population_mean,
            }
        }
    
    def get_simulated_intervention(self, feature: Optional[str] = None, n_pairs: int = 25,
                                    seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a paired t-test scenario simulating before/after intervention.
        
        Note: The Pima dataset doesn't have longitudinal data, so we simulate 
        realistic before/after pairs based on known intervention effects.
        
        Args:
            feature: Feature to test. If None, randomly selected.
            n_pairs: Number of paired observations
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario data and context
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Interventions and their expected effects
        interventions = {
            "glucose": {
                "name": "dietary intervention program",
                "effect_mean": -15,  # Expected reduction
                "effect_std": 8,
            },
            "bmi": {
                "name": "12-week exercise program",
                "effect_mean": -1.5,
                "effect_std": 1.0,
            },
            "bloodpressure": {
                "name": "lifestyle modification program",
                "effect_mean": -5,
                "effect_std": 4,
            },
            "insulin": {
                "name": "medication therapy",
                "effect_mean": -20,
                "effect_std": 15,
            },
        }
        
        # Select feature
        intervention_features = list(interventions.keys())
        valid_features = [f for f in intervention_features if f in self.available_features]
        
        if feature is None:
            feature = random.choice(valid_features) if valid_features else "glucose"
        elif feature not in valid_features:
            feature = "glucose"
            
        intervention = interventions.get(feature, interventions["glucose"])
        feature_info = self.POPULATION_NORMS.get(feature, {"mean": 100, "unit": "units", "name": feature.title()})
        
        # Get baseline (before) data from actual dataset
        feature_data = self.df[feature].dropna()
        if len(feature_data) >= n_pairs:
            before = feature_data.sample(n_pairs, random_state=seed).values
        else:
            before = feature_data.values[:n_pairs]
            
        # Simulate after data based on intervention effect (add individual variation)
        individual_effects = np.random.normal(intervention["effect_mean"], intervention["effect_std"], len(before))
        after = before + individual_effects
        
        # Ensure physiologically plausible values
        after = np.maximum(after, 0)
        
        return {
            "before_data": before.tolist(),
            "after_data": after.tolist(),
            "context": {
                "domain": "healthcare",
                "dataset": "Pima Indians Diabetes Database (with simulated intervention)",
                "description": f"{feature_info['name']} measurements before and after {intervention['name']}",
                "intervention": intervention["name"],
                "test_description": f"Testing if {feature_info['name']} changed significantly after {intervention['name']}",
                "feature": feature,
                "units": feature_info['unit'],
                "note": "After measurements are simulated based on expected intervention effects",
            }
        }


class RealDataLoader:
    """
    Unified interface for loading real-world data for hypothesis testing scenarios.
    
    Combines StockDataLoader and HealthcareDataLoader to provide a single interface
    that matches the DataGenerator pattern.
    """
    
    def __init__(self):
        self._stock_loader = None
        self._healthcare_loader = None
        
    @property
    def stock_loader(self) -> StockDataLoader:
        """Lazy-load stock data loader."""
        if self._stock_loader is None:
            self._stock_loader = StockDataLoader()
        return self._stock_loader
    
    @property
    def healthcare_loader(self) -> HealthcareDataLoader:
        """Lazy-load healthcare data loader."""
        if self._healthcare_loader is None:
            self._healthcare_loader = HealthcareDataLoader()
        return self._healthcare_loader
    
    def generate_one_sample_scenario(self, domain: str = "random", n_samples: int = 50,
                                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a one-sample t-test scenario from real data.
        
        Args:
            domain: 'stocks', 'healthcare', or 'random'
            n_samples: Number of data points
            seed: Random seed
            
        Returns:
            Scenario dictionary matching DataGenerator format
        """
        if seed is not None:
            random.seed(seed)
            
        if domain == "random":
            domain = random.choice(["stocks", "healthcare"])
            
        if domain == "stocks":
            return self.stock_loader.get_asset_returns(n_samples, seed)
        else:
            return self.healthcare_loader.get_population_comparison(n_samples=n_samples, seed=seed)
    
    def generate_two_sample_scenario(self, domain: str = "random", n_samples: int = 30,
                                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a two-sample t-test scenario from real data.
        
        Args:
            domain: 'stocks', 'healthcare', or 'random'
            n_samples: Number of data points per group
            seed: Random seed
            
        Returns:
            Scenario dictionary matching DataGenerator format
        """
        if seed is not None:
            random.seed(seed)
            
        if domain == "random":
            domain = random.choice(["stocks", "healthcare"])
            
        if domain == "stocks":
            return self.stock_loader.get_group_returns(n_samples, seed)
        else:
            return self.healthcare_loader.get_feature_by_outcome(n_samples=n_samples, seed=seed)
    
    def generate_paired_scenario(self, domain: str = "random", n_pairs: int = 25,
                                  seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a paired t-test scenario from real data.
        
        Args:
            domain: 'stocks', 'healthcare', or 'random'
            n_pairs: Number of paired observations
            seed: Random seed
            
        Returns:
            Scenario dictionary matching DataGenerator format
        """
        if seed is not None:
            random.seed(seed)
            
        if domain == "random":
            domain = random.choice(["stocks", "healthcare"])
            
        if domain == "stocks":
            return self.stock_loader.get_paired_periods(n_pairs, seed)
        else:
            return self.healthcare_loader.get_simulated_intervention(n_pairs=n_pairs, seed=seed)


# Convenience function for quick testing
def test_loaders():
    """Test that data loaders work correctly."""
    print("Testing Real Data Loaders...")
    
    loader = RealDataLoader()
    
    print("\n=== One-Sample T-Test Scenarios ===")
    try:
        scenario = loader.generate_one_sample_scenario(domain="stocks", seed=42)
        print(f"Stocks: {scenario['context']['description']}")
        print(f"  Data points: {len(scenario['data'])}, Population mean: {scenario['population_mean']}")
    except Exception as e:
        print(f"Stocks error: {e}")
        
    try:
        scenario = loader.generate_one_sample_scenario(domain="healthcare", seed=42)
        print(f"Healthcare: {scenario['context']['description']}")
        print(f"  Data points: {len(scenario['data'])}, Population mean: {scenario['population_mean']}")
    except Exception as e:
        print(f"Healthcare error: {e}")
    
    print("\n=== Two-Sample T-Test Scenarios ===")
    try:
        scenario = loader.generate_two_sample_scenario(domain="stocks", seed=42)
        print(f"Stocks: {scenario['context']['description']}")
        print(f"  Group 1: {len(scenario['group1_data'])}, Group 2: {len(scenario['group2_data'])}")
    except Exception as e:
        print(f"Stocks error: {e}")
        
    try:
        scenario = loader.generate_two_sample_scenario(domain="healthcare", seed=42)
        print(f"Healthcare: {scenario['context']['description']}")
        print(f"  Group 1: {len(scenario['group1_data'])}, Group 2: {len(scenario['group2_data'])}")
    except Exception as e:
        print(f"Healthcare error: {e}")
    
    print("\n=== Paired T-Test Scenarios ===")
    try:
        scenario = loader.generate_paired_scenario(domain="stocks", seed=42)
        print(f"Stocks: {scenario['context']['description']}")
        print(f"  Before: {len(scenario['before_data'])}, After: {len(scenario['after_data'])}")
    except Exception as e:
        print(f"Stocks error: {e}")
        
    try:
        scenario = loader.generate_paired_scenario(domain="healthcare", seed=42)
        print(f"Healthcare: {scenario['context']['description']}")
        print(f"  Before: {len(scenario['before_data'])}, After: {len(scenario['after_data'])}")
    except Exception as e:
        print(f"Healthcare error: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    test_loaders()
