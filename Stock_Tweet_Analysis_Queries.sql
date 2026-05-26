-- ============================================================================
-- STOCK TWEET SENTIMENT ANALYSIS: SQL QUERY REFERENCE (MySQL 8.0+)
-- ============================================================================
-- Project: Twitter Sentiment vs. Stock Price Movements
-- Dataset: Kaggle - Stock Tweets for Sentiment Analysis and Prediction
--          https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction
-- Tables:  tweets (cleaned_tweets_scored.csv)
--          stock_prices (stock_yfinance_data.csv)
-- ============================================================================


-- 0-1: Create the tweets table from the cleaned CSV
CREATE TABLE tweets (
    tweet_date      DATETIME,
    tweet_text      TEXT,
    stock_name      VARCHAR(10),
    company_name    VARCHAR(100),
    vader_score     DECIMAL(6,4),
    textblob_score  DECIMAL(6,4),
    transformer_score DECIMAL(6,4)
);

-- 0-2: Create the stock prices table from Yahoo Finance CSV
CREATE TABLE stock_prices (
    trade_date  DATE,
    open_price  DECIMAL(10,4),
    high_price  DECIMAL(10,4),
    low_price   DECIMAL(10,4),
    close_price DECIMAL(10,4),
    adj_close   DECIMAL(10,4),
    volume      BIGINT,
    stock_name  VARCHAR(10)
);

-- 0-3: Compute daily returns and next-day returns
-- Feeds: All analysis slides
CREATE VIEW stock_returns AS
SELECT
    stock_name,
    trade_date,
    adj_close,
    volume,
    ((adj_close / LAG(adj_close) OVER (PARTITION BY stock_name ORDER BY trade_date)) - 1) * 100
        AS daily_return,
    LEAD(
        ((adj_close / LAG(adj_close) OVER (PARTITION BY stock_name ORDER BY trade_date)) - 1) * 100
    ) OVER (PARTITION BY stock_name ORDER BY trade_date)
        AS next_day_return,
    ABS(((adj_close / LAG(adj_close) OVER (PARTITION BY stock_name ORDER BY trade_date)) - 1) * 100)
        AS volatility
FROM stock_prices;

-- 0-4: Aggregate tweets to daily level per ticker
-- Feeds: Slide 6 (Data Aggregation Pipeline)
CREATE VIEW daily_sentiment AS
SELECT
    stock_name,
    DATE(tweet_date) AS trade_date,
    COUNT(*)         AS tweet_count,
    AVG(vader_score)       AS vader_mean,
    AVG(textblob_score)    AS textblob_mean,
    AVG(transformer_score) AS transformer_mean,
    STDDEV(transformer_score) AS transformer_std
FROM tweets
GROUP BY stock_name, DATE(tweet_date);

-- 0-5: Merge sentiment with price data
-- Feeds: All analysis slides
CREATE VIEW merged_analysis AS
SELECT
    ds.stock_name,
    ds.trade_date,
    ds.tweet_count,
    ds.vader_mean,
    ds.textblob_mean,
    ds.transformer_mean,
    ds.transformer_std,
    sr.daily_return,
    sr.next_day_return,
    sr.volume AS trading_volume,
    sr.volatility,
    sr.adj_close
FROM daily_sentiment ds
INNER JOIN stock_returns sr
    ON ds.stock_name = sr.stock_name
   AND ds.trade_date = sr.trade_date
WHERE sr.daily_return IS NOT NULL;


-- ============================================================================
-- SECTION 1: DATASET OVERVIEW STATISTICS
-- Feeds: Slides 3-4 (Dataset Overview, Aggregation Pipeline)
-- ============================================================================

-- 1-1: Overall dataset dimensions
SELECT 'tweets' AS dataset, COUNT(*) AS row_count FROM tweets
UNION ALL
SELECT 'stock_prices', COUNT(*) FROM stock_prices
UNION ALL
SELECT 'merged_daily', COUNT(*) FROM merged_analysis;

-- 1-2: Tweet count by ticker (top 10)
SELECT
    stock_name,
    company_name,
    COUNT(*) AS tweet_count,
    ROUND(AVG(transformer_score), 3) AS avg_transformer,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM tweets), 1) AS pct_of_total
FROM tweets
GROUP BY stock_name, company_name
ORDER BY tweet_count DESC
LIMIT 10;

-- 1-3: Date range and trading day count
SELECT
    MIN(trade_date) AS start_date,
    MAX(trade_date) AS end_date,
    COUNT(DISTINCT trade_date) AS trading_days
FROM stock_prices;

-- 1-4: Sentiment score distributions
SELECT
    'VADER'       AS model,
    ROUND(AVG(vader_score), 4) AS mean_score,
    ROUND(STDDEV(vader_score), 4) AS std_dev,
    MIN(vader_score) AS min_val,
    MAX(vader_score) AS max_val
FROM tweets
UNION ALL
SELECT
    'TextBlob',
    ROUND(AVG(textblob_score), 4),
    ROUND(STDDEV(textblob_score), 4),
    MIN(textblob_score),
    MAX(textblob_score)
FROM tweets
UNION ALL
SELECT
    'Transformer',
    ROUND(AVG(transformer_score), 4),
    ROUND(STDDEV(transformer_score), 4),
    MIN(transformer_score),
    MAX(transformer_score)
FROM tweets;


-- ============================================================================
-- SECTION 2: COMPREHENSIVE CORRELATION TABLE
-- Feeds: Slide 8 (Correlation Analysis)
-- ============================================================================
-- NOTE: MySQL has no built-in CORR() function.
-- Pearson r = (N*SUM(x*y) - SUM(x)*SUM(y)) /
--             (SQRT(N*SUM(x*x) - SUM(x)*SUM(x)) * SQRT(N*SUM(y*y) - SUM(y)*SUM(y)))
-- ============================================================================

-- 2-1: Transformer vs same-day return
SELECT
    'Transformer vs Same-Day' AS relationship,
    COUNT(*) AS n,
    ROUND(
        (COUNT(*) * SUM(transformer_mean * daily_return) - SUM(transformer_mean) * SUM(daily_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(daily_return * daily_return) - POW(SUM(daily_return), 2)))
    , 4) AS pearson_r
FROM merged_analysis
WHERE daily_return IS NOT NULL;

-- 2-2: Transformer vs next-day return
SELECT
    'Transformer vs Next-Day' AS relationship,
    COUNT(*) AS n,
    ROUND(
        (COUNT(*) * SUM(transformer_mean * next_day_return) - SUM(transformer_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4) AS pearson_r
FROM merged_analysis
WHERE next_day_return IS NOT NULL;

-- 2-3: VADER vs same-day and next-day return
SELECT
    'VADER vs Same-Day' AS relationship,
    ROUND(
        (COUNT(*) * SUM(vader_mean * daily_return) - SUM(vader_mean) * SUM(daily_return))
        /
        (SQRT(COUNT(*) * SUM(vader_mean * vader_mean) - POW(SUM(vader_mean), 2))
         * SQRT(COUNT(*) * SUM(daily_return * daily_return) - POW(SUM(daily_return), 2)))
    , 4) AS pearson_r
FROM merged_analysis
WHERE daily_return IS NOT NULL
UNION ALL
SELECT
    'VADER vs Next-Day',
    ROUND(
        (COUNT(*) * SUM(vader_mean * next_day_return) - SUM(vader_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(vader_mean * vader_mean) - POW(SUM(vader_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4)
FROM merged_analysis
WHERE next_day_return IS NOT NULL;

-- 2-4: TextBlob vs same-day and next-day return
SELECT
    'TextBlob vs Same-Day' AS relationship,
    ROUND(
        (COUNT(*) * SUM(textblob_mean * daily_return) - SUM(textblob_mean) * SUM(daily_return))
        /
        (SQRT(COUNT(*) * SUM(textblob_mean * textblob_mean) - POW(SUM(textblob_mean), 2))
         * SQRT(COUNT(*) * SUM(daily_return * daily_return) - POW(SUM(daily_return), 2)))
    , 4) AS pearson_r
FROM merged_analysis
WHERE daily_return IS NOT NULL
UNION ALL
SELECT
    'TextBlob vs Next-Day',
    ROUND(
        (COUNT(*) * SUM(textblob_mean * next_day_return) - SUM(textblob_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(textblob_mean * textblob_mean) - POW(SUM(textblob_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4)
FROM merged_analysis
WHERE next_day_return IS NOT NULL;

-- 2-5: Tweet volume vs trading volume and volatility
SELECT
    'Tweet Vol vs Trading Vol' AS relationship,
    ROUND(
        (COUNT(*) * SUM(tweet_count * trading_volume) - SUM(tweet_count) * SUM(trading_volume))
        /
        (SQRT(COUNT(*) * SUM(tweet_count * tweet_count) - POW(SUM(tweet_count), 2))
         * SQRT(COUNT(*) * SUM(trading_volume * trading_volume) - POW(SUM(trading_volume), 2)))
    , 4) AS pearson_r
FROM merged_analysis
UNION ALL
SELECT
    'Tweet Vol vs Volatility',
    ROUND(
        (COUNT(*) * SUM(tweet_count * volatility) - SUM(tweet_count) * SUM(volatility))
        /
        (SQRT(COUNT(*) * SUM(tweet_count * tweet_count) - POW(SUM(tweet_count), 2))
         * SQRT(COUNT(*) * SUM(volatility * volatility) - POW(SUM(volatility), 2)))
    , 4)
FROM merged_analysis;


-- ============================================================================
-- SECTION 3: Q1 - DO POSITIVE SENTIMENT DAYS PRECEDE POSITIVE RETURNS?
-- Feeds: Slides 9-10 (Q1 Analysis)
-- ============================================================================

-- 3-1: Bucket transformer sentiment and compute next-day return stats
WITH bucketed AS (
    SELECT
        next_day_return,
        CASE
            WHEN transformer_mean < -0.5 THEN 'Very Negative'
            WHEN transformer_mean < -0.1 THEN 'Negative'
            WHEN transformer_mean <= 0.1  THEN 'Neutral'
            WHEN transformer_mean <= 0.5  THEN 'Positive'
            ELSE 'Very Positive'
        END AS sentiment_bucket
    FROM merged_analysis
    WHERE next_day_return IS NOT NULL
)
SELECT
    sentiment_bucket,
    COUNT(*) AS n,
    ROUND(AVG(next_day_return), 3) AS mean_return_pct,
    ROUND(STDDEV(next_day_return), 3) AS std_return,
    ROUND(AVG(next_day_return) - 1.96 * STDDEV(next_day_return) / SQRT(COUNT(*)), 3) AS ci_lower,
    ROUND(AVG(next_day_return) + 1.96 * STDDEV(next_day_return) / SQRT(COUNT(*)), 3) AS ci_upper
FROM bucketed
GROUP BY sentiment_bucket
ORDER BY
    CASE sentiment_bucket
        WHEN 'Very Negative' THEN 1
        WHEN 'Negative' THEN 2
        WHEN 'Neutral' THEN 3
        WHEN 'Positive' THEN 4
        WHEN 'Very Positive' THEN 5
    END;


-- ============================================================================
-- SECTION 4: Q2 - WHICH STOCKS ARE MOST SENSITIVE TO SENTIMENT?
-- Feeds: Slides 11-12 (Q2 Analysis, Directional Accuracy)
-- ============================================================================

-- 4-1: Per-stock Pearson correlation (Transformer vs next-day return)
SELECT
    stock_name,
    COUNT(*) AS n,
    ROUND(
        (COUNT(*) * SUM(transformer_mean * next_day_return) - SUM(transformer_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4) AS pearson_r
FROM merged_analysis
WHERE next_day_return IS NOT NULL
GROUP BY stock_name
HAVING COUNT(*) > 20
ORDER BY ABS(
    (COUNT(*) * SUM(transformer_mean * next_day_return) - SUM(transformer_mean) * SUM(next_day_return))
    /
    (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
     * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
) DESC;

-- 4-2: Directional accuracy by stock (Transformer model)
WITH directions AS (
    SELECT
        stock_name,
        CASE
            WHEN (transformer_mean > 0 AND next_day_return > 0)
              OR (transformer_mean < 0 AND next_day_return < 0)
            THEN 1 ELSE 0
        END AS correct_direction
    FROM merged_analysis
    WHERE next_day_return IS NOT NULL
      AND transformer_mean != 0
)
SELECT
    stock_name,
    COUNT(*) AS n,
    ROUND(100.0 * SUM(correct_direction) / COUNT(*), 1) AS accuracy_pct
FROM directions
GROUP BY stock_name
HAVING COUNT(*) > 20
ORDER BY accuracy_pct DESC;

-- 4-3: Directional accuracy comparison across all three models
WITH model_accuracy AS (
    SELECT
        CASE WHEN (transformer_mean > 0 AND next_day_return > 0)
              OR (transformer_mean < 0 AND next_day_return < 0) THEN 1 ELSE 0 END AS trans_correct,
        CASE WHEN (vader_mean > 0 AND next_day_return > 0)
              OR (vader_mean < 0 AND next_day_return < 0) THEN 1 ELSE 0 END AS vader_correct,
        CASE WHEN (textblob_mean > 0 AND next_day_return > 0)
              OR (textblob_mean < 0 AND next_day_return < 0) THEN 1 ELSE 0 END AS textblob_correct
    FROM merged_analysis
    WHERE next_day_return IS NOT NULL
)
SELECT
    'Transformer' AS model,
    ROUND(100.0 * AVG(trans_correct), 1) AS directional_accuracy_pct
FROM model_accuracy
UNION ALL
SELECT 'VADER', ROUND(100.0 * AVG(vader_correct), 1) FROM model_accuracy
UNION ALL
SELECT 'TextBlob', ROUND(100.0 * AVG(textblob_correct), 1) FROM model_accuracy;


-- ============================================================================
-- SECTION 5: Q3 - IS TWEET VOLUME A BETTER PREDICTOR THAN SENTIMENT?
-- Feeds: Slides 13-14 (Q3 Volume Analysis)
-- ============================================================================

-- 5-1: Volume quartile analysis
WITH quartiled AS (
    SELECT
        tweet_count,
        daily_return,
        next_day_return,
        volatility,
        trading_volume,
        NTILE(4) OVER (ORDER BY tweet_count) AS vol_quartile
    FROM merged_analysis
    WHERE next_day_return IS NOT NULL
)
SELECT
    vol_quartile AS quartile,
    COUNT(*) AS n,
    ROUND(AVG(tweet_count), 1) AS avg_tweets,
    ROUND(AVG(volatility), 3) AS avg_abs_return,
    ROUND(AVG(trading_volume), 0) AS avg_trading_vol
FROM quartiled
GROUP BY vol_quartile
ORDER BY vol_quartile;

-- 5-2: Correlation comparison: volume metrics vs sentiment metrics
SELECT
    'Tweet Vol vs Trading Vol' AS metric_pair,
    ROUND(
        (COUNT(*) * SUM(tweet_count * trading_volume) - SUM(tweet_count) * SUM(trading_volume))
        /
        (SQRT(COUNT(*) * SUM(tweet_count * tweet_count) - POW(SUM(tweet_count), 2))
         * SQRT(COUNT(*) * SUM(trading_volume * trading_volume) - POW(SUM(trading_volume), 2)))
    , 4) AS correlation
FROM merged_analysis
UNION ALL
SELECT 'Tweet Vol vs Volatility',
    ROUND(
        (COUNT(*) * SUM(tweet_count * volatility) - SUM(tweet_count) * SUM(volatility))
        /
        (SQRT(COUNT(*) * SUM(tweet_count * tweet_count) - POW(SUM(tweet_count), 2))
         * SQRT(COUNT(*) * SUM(volatility * volatility) - POW(SUM(volatility), 2)))
    , 4)
FROM merged_analysis
UNION ALL
SELECT 'Transformer vs Next-Day Return',
    ROUND(
        (COUNT(*) * SUM(transformer_mean * next_day_return) - SUM(transformer_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4)
FROM merged_analysis
WHERE next_day_return IS NOT NULL
UNION ALL
SELECT 'VADER vs Next-Day Return',
    ROUND(
        (COUNT(*) * SUM(vader_mean * next_day_return) - SUM(vader_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(vader_mean * vader_mean) - POW(SUM(vader_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4)
FROM merged_analysis
WHERE next_day_return IS NOT NULL;


-- ============================================================================
-- SECTION 6: Q4 - HOW QUICKLY DOES SENTIMENT SHIFT AFTER LARGE MOVES?
-- Feeds: Slides 15-16 (Q4 Earnings/Event Analysis)
-- ============================================================================

-- 6-1: Sentiment change following large price moves (proxy for earnings events)
WITH lagged AS (
    SELECT
        stock_name,
        trade_date,
        transformer_mean,
        daily_return,
        LAG(daily_return) OVER (PARTITION BY stock_name ORDER BY trade_date) AS prev_return,
        LAG(transformer_mean) OVER (PARTITION BY stock_name ORDER BY trade_date) AS prev_sentiment
    FROM merged_analysis
)
SELECT
    CASE
        WHEN prev_return > 3 THEN 'After +3% Day'
        WHEN prev_return < -3 THEN 'After -3% Day'
        ELSE 'Normal Day'
    END AS event_type,
    COUNT(*) AS n,
    ROUND(AVG(transformer_mean - prev_sentiment), 4) AS avg_sentiment_change,
    ROUND(AVG(transformer_mean), 4) AS avg_sentiment_after,
    ROUND(AVG(daily_return), 3) AS avg_return_after
FROM lagged
WHERE prev_return IS NOT NULL
  AND prev_sentiment IS NOT NULL
GROUP BY
    CASE
        WHEN prev_return > 3 THEN 'After +3% Day'
        WHEN prev_return < -3 THEN 'After -3% Day'
        ELSE 'Normal Day'
    END
ORDER BY event_type;


-- ============================================================================
-- SECTION 7: Q5 - CAN NEGATIVE SENTIMENT SPIKES PREDICT SELL-OFFS?
-- Feeds: Slides 17-18 (Q5 Spike Analysis)
-- ============================================================================

-- 7-1: Identify negative sentiment spikes (bottom 10th percentile)
-- MySQL does not support PERCENTILE_CONT; use a ranked subquery instead.
WITH ranked AS (
    SELECT
        transformer_mean,
        next_day_return,
        ROW_NUMBER() OVER (ORDER BY transformer_mean) AS rn,
        COUNT(*) OVER () AS total_rows
    FROM merged_analysis
    WHERE next_day_return IS NOT NULL
),
threshold AS (
    SELECT transformer_mean AS p10
    FROM ranked
    WHERE rn = FLOOR(0.10 * total_rows) + 1
    LIMIT 1
),
classified AS (
    SELECT
        r.next_day_return,
        CASE WHEN r.transformer_mean <= t.p10 THEN 'Negative Spike' ELSE 'Non-Spike' END AS spike_flag
    FROM ranked r
    CROSS JOIN threshold t
)
SELECT
    spike_flag,
    COUNT(*) AS n,
    ROUND(AVG(next_day_return), 4) AS mean_next_return,
    ROUND(100.0 * SUM(CASE WHEN next_day_return < 0 THEN 1 ELSE 0 END) / COUNT(*), 1)
        AS pct_negative_next_day
FROM classified
GROUP BY spike_flag;


-- ============================================================================
-- SECTION 8: Q6 - DOES SENTIMENT DISPERSION PREDICT VOLATILITY?
-- Feeds: Slides 19-20 (Additional Q6 Analysis)
-- ============================================================================

-- 8-1: Correlation between intraday sentiment dispersion and realized volatility
SELECT
    'Transformer StdDev vs Volatility' AS relationship,
    COUNT(*) AS n,
    ROUND(
        (COUNT(*) * SUM(transformer_std * volatility) - SUM(transformer_std) * SUM(volatility))
        /
        (SQRT(COUNT(*) * SUM(transformer_std * transformer_std) - POW(SUM(transformer_std), 2))
         * SQRT(COUNT(*) * SUM(volatility * volatility) - POW(SUM(volatility), 2)))
    , 4) AS pearson_r
FROM merged_analysis
WHERE transformer_std IS NOT NULL
  AND transformer_std > 0;

-- 8-2: Dispersion quartile breakdown
WITH disp_q AS (
    SELECT
        transformer_std,
        volatility,
        trading_volume,
        NTILE(4) OVER (ORDER BY transformer_std) AS disp_quartile
    FROM merged_analysis
    WHERE transformer_std IS NOT NULL AND transformer_std > 0
)
SELECT
    disp_quartile,
    COUNT(*) AS n,
    ROUND(AVG(transformer_std), 4) AS avg_dispersion,
    ROUND(AVG(volatility), 3) AS avg_volatility,
    ROUND(AVG(trading_volume), 0) AS avg_trading_vol
FROM disp_q
GROUP BY disp_quartile
ORDER BY disp_quartile;


-- ============================================================================
-- SECTION 9: Q7 - HIGH VS LOW TWEET-VOLUME TICKERS
-- Feeds: Slides 21-22 (Additional Q7 Analysis)
-- ============================================================================

-- 9-1: Classify tickers into high-volume and low-volume tiers
-- MySQL does not support PERCENTILE_CONT; use a ranked subquery for the median.
WITH ticker_volume AS (
    SELECT
        stock_name,
        COUNT(*) AS total_tweets
    FROM tweets
    GROUP BY stock_name
),
ranked_tv AS (
    SELECT
        stock_name,
        total_tweets,
        ROW_NUMBER() OVER (ORDER BY total_tweets) AS rn,
        COUNT(*) OVER () AS total_rows
    FROM ticker_volume
),
median_val AS (
    SELECT total_tweets AS median_tweets
    FROM ranked_tv
    WHERE rn = FLOOR(0.50 * total_rows) + 1
    LIMIT 1
),
tier AS (
    SELECT
        tv.stock_name,
        tv.total_tweets,
        CASE
            WHEN tv.total_tweets >= mv.median_tweets THEN 'High Volume'
            ELSE 'Low Volume'
        END AS volume_tier
    FROM ticker_volume tv
    CROSS JOIN median_val mv
)
SELECT
    t.volume_tier,
    COUNT(*) AS n_observations,
    ROUND(
        (COUNT(*) * SUM(m.transformer_mean * m.next_day_return) - SUM(m.transformer_mean) * SUM(m.next_day_return))
        /
        (SQRT(COUNT(*) * SUM(m.transformer_mean * m.transformer_mean) - POW(SUM(m.transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(m.next_day_return * m.next_day_return) - POW(SUM(m.next_day_return), 2)))
    , 4) AS pearson_r,
    ROUND(100.0 * SUM(CASE
        WHEN (m.transformer_mean > 0 AND m.next_day_return > 0)
          OR (m.transformer_mean < 0 AND m.next_day_return < 0)
        THEN 1 ELSE 0 END) / COUNT(*), 1) AS directional_accuracy
FROM merged_analysis m
INNER JOIN tier t ON m.stock_name = t.stock_name
WHERE m.next_day_return IS NOT NULL
GROUP BY t.volume_tier;


-- ============================================================================
-- SECTION 10: MODEL COMPARISON AND DIVERGENCE
-- Feeds: Slide 23 (Model Divergence Analysis)
-- ============================================================================

-- 10-1: Model-level summary statistics
SELECT
    'VADER' AS model,
    ROUND(AVG(vader_mean), 4) AS mean_score,
    ROUND(
        (COUNT(*) * SUM(vader_mean * daily_return) - SUM(vader_mean) * SUM(daily_return))
        /
        (SQRT(COUNT(*) * SUM(vader_mean * vader_mean) - POW(SUM(vader_mean), 2))
         * SQRT(COUNT(*) * SUM(daily_return * daily_return) - POW(SUM(daily_return), 2)))
    , 4) AS same_day_r,
    ROUND(
        (COUNT(*) * SUM(vader_mean * next_day_return) - SUM(vader_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(vader_mean * vader_mean) - POW(SUM(vader_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4) AS next_day_r
FROM merged_analysis
WHERE next_day_return IS NOT NULL
UNION ALL
SELECT 'TextBlob',
    ROUND(AVG(textblob_mean), 4),
    ROUND(
        (COUNT(*) * SUM(textblob_mean * daily_return) - SUM(textblob_mean) * SUM(daily_return))
        /
        (SQRT(COUNT(*) * SUM(textblob_mean * textblob_mean) - POW(SUM(textblob_mean), 2))
         * SQRT(COUNT(*) * SUM(daily_return * daily_return) - POW(SUM(daily_return), 2)))
    , 4),
    ROUND(
        (COUNT(*) * SUM(textblob_mean * next_day_return) - SUM(textblob_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(textblob_mean * textblob_mean) - POW(SUM(textblob_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4)
FROM merged_analysis
WHERE next_day_return IS NOT NULL
UNION ALL
SELECT 'Transformer',
    ROUND(AVG(transformer_mean), 4),
    ROUND(
        (COUNT(*) * SUM(transformer_mean * daily_return) - SUM(transformer_mean) * SUM(daily_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(daily_return * daily_return) - POW(SUM(daily_return), 2)))
    , 4),
    ROUND(
        (COUNT(*) * SUM(transformer_mean * next_day_return) - SUM(transformer_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4)
FROM merged_analysis
WHERE next_day_return IS NOT NULL;

-- 10-2: Average absolute divergence between models
SELECT
    ROUND(AVG(ABS(vader_mean - textblob_mean)), 3) AS vader_vs_textblob,
    ROUND(AVG(ABS(vader_mean - transformer_mean)), 3) AS vader_vs_transformer,
    ROUND(AVG(ABS(textblob_mean - transformer_mean)), 3) AS textblob_vs_transformer
FROM merged_analysis;


-- ============================================================================
-- SECTION 11: SUPPLEMENTARY QUERIES FOR APPENDIX
-- Feeds: Appendix slides
-- ============================================================================

-- 11-1: Daily tweet count distribution
SELECT
    MIN(tweet_count) AS min_tweets,
    ROUND(AVG(tweet_count), 1) AS avg_tweets,
    MAX(tweet_count) AS max_tweets,
    ROUND(STDDEV(tweet_count), 1) AS std_tweets
FROM merged_analysis;

-- 11-2: Per-ticker summary for appendix table
SELECT
    stock_name,
    COUNT(*) AS trading_days,
    ROUND(AVG(tweet_count), 1) AS avg_daily_tweets,
    ROUND(AVG(transformer_mean), 4) AS avg_transformer,
    ROUND(
        (COUNT(*) * SUM(transformer_mean * next_day_return) - SUM(transformer_mean) * SUM(next_day_return))
        /
        (SQRT(COUNT(*) * SUM(transformer_mean * transformer_mean) - POW(SUM(transformer_mean), 2))
         * SQRT(COUNT(*) * SUM(next_day_return * next_day_return) - POW(SUM(next_day_return), 2)))
    , 4) AS next_day_corr,
    ROUND(AVG(volatility), 3) AS avg_volatility
FROM merged_analysis
WHERE next_day_return IS NOT NULL
GROUP BY stock_name
ORDER BY stock_name;

-- End of queries
