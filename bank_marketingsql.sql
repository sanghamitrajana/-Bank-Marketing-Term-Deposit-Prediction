-- =====================================================================
-- BANK MARKETING: SQL ANALYSIS LAYER (MySQL 8.0+ version)
-- Order of use:
--   1. Run the SETUP block below (creates the database and two empty tables)
--   2. Import bank_marketing_clean.csv into bank_marketing and
--      lead_scores.csv into lead_scores (Table Data Import Wizard)
--   3. Run the queries one at a time (select a statement, press Ctrl+Enter)
-- =====================================================================


-- =====================================================================
-- SETUP
-- =====================================================================

-- S.1 Create and select the database
CREATE DATABASE IF NOT EXISTS bank_marketing_db;


-- S.2 Select the database for the rest of the session
USE bank_marketing_db;


-- S.3 Main table: one row per client contact (column `default` renamed credit_default)
CREATE TABLE IF NOT EXISTS bank_marketing (
    client_id      INT PRIMARY KEY,
    age            INT,
    job            VARCHAR(20),
    marital        VARCHAR(15),
    education      VARCHAR(15),
    credit_default VARCHAR(3),
    balance        INT,
    housing        VARCHAR(3),
    loan           VARCHAR(3),
    contact        VARCHAR(15),
    day_of_week    INT,
    month          VARCHAR(3),
    duration       INT,
    campaign       INT,
    pdays          INT,
    previous       INT,
    poutcome       VARCHAR(10),
    y              VARCHAR(3)
);


-- S.4 Model output table: test-set scores from the XGBoost model
CREATE TABLE IF NOT EXISTS lead_scores (
    client_id  INT PRIMARY KEY,
    prob_yes   DOUBLE,
    actual_yes TINYINT
);
SELECT COUNT(*) FROM bank_marketing;                 -- expect 45211
SELECT y, COUNT(*) FROM bank_marketing GROUP BY y;   -- no 39922, yes 5289

-- =====================================================================
-- SECTION 1: DATA QUALITY CHECKS
-- =====================================================================

-- 1.1 Row count
SELECT COUNT(*) AS total_rows
FROM bank_marketing;


-- 1.2 Duplicate check: groups of identical records (0 expected)
SELECT COUNT(*) AS duplicate_groups
FROM (
    SELECT age, job, marital, education, credit_default, balance, housing, loan,
           contact, day_of_week, month, duration, campaign, pdays, previous,
           poutcome, y
    FROM bank_marketing
    GROUP BY age, job, marital, education, credit_default, balance, housing, loan,
             contact, day_of_week, month, duration, campaign, pdays, previous,
             poutcome, y
    HAVING COUNT(*) > 1
) AS dup;


-- 1.3 NULL count per column (0 expected after unknown values were restored)
SELECT
    SUM(CASE WHEN job       IS NULL THEN 1 ELSE 0 END) AS null_job,
    SUM(CASE WHEN education IS NULL THEN 1 ELSE 0 END) AS null_education,
    SUM(CASE WHEN contact   IS NULL THEN 1 ELSE 0 END) AS null_contact,
    SUM(CASE WHEN poutcome  IS NULL THEN 1 ELSE 0 END) AS null_poutcome,
    SUM(CASE WHEN balance   IS NULL THEN 1 ELSE 0 END) AS null_balance,
    SUM(CASE WHEN y         IS NULL THEN 1 ELSE 0 END) AS null_target
FROM bank_marketing;


-- 1.4 'unknown' as an explicit category: count and share per column
SELECT 'job' AS column_name,
       SUM(CASE WHEN job = 'unknown' THEN 1 ELSE 0 END) AS unknown_rows,
       ROUND(100.0 * SUM(CASE WHEN job = 'unknown' THEN 1 ELSE 0 END) / COUNT(*), 2) AS unknown_pct
FROM bank_marketing
UNION ALL
SELECT 'education',
       SUM(CASE WHEN education = 'unknown' THEN 1 ELSE 0 END),
       ROUND(100.0 * SUM(CASE WHEN education = 'unknown' THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM bank_marketing
UNION ALL
SELECT 'contact',
       SUM(CASE WHEN contact = 'unknown' THEN 1 ELSE 0 END),
       ROUND(100.0 * SUM(CASE WHEN contact = 'unknown' THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM bank_marketing
UNION ALL
SELECT 'poutcome',
       SUM(CASE WHEN poutcome = 'unknown' THEN 1 ELSE 0 END),
       ROUND(100.0 * SUM(CASE WHEN poutcome = 'unknown' THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM bank_marketing;


-- 1.5 Rule-based validation checklist (failed_rows should be 0 unless noted)
SELECT 'age outside 18-100' AS check_name, COUNT(*) AS failed_rows
FROM bank_marketing WHERE age < 18 OR age > 100
UNION ALL
SELECT 'campaign < 1', COUNT(*)
FROM bank_marketing WHERE campaign < 1
UNION ALL
SELECT 'negative duration', COUNT(*)
FROM bank_marketing WHERE duration < 0
UNION ALL
SELECT 'day outside 1-31', COUNT(*)
FROM bank_marketing WHERE day_of_week < 1 OR day_of_week > 31
UNION ALL
SELECT 'invalid target value', COUNT(*)
FROM bank_marketing WHERE y NOT IN ('yes', 'no')
UNION ALL
SELECT 'invalid month label', COUNT(*)
FROM bank_marketing
WHERE month NOT IN ('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec')
UNION ALL
SELECT 'pdays=-1 disagrees with previous=0 (cross-field consistency)', COUNT(*)
FROM bank_marketing
WHERE (pdays = -1 AND previous <> 0) OR (pdays <> -1 AND previous = 0);


-- 1.6 Sentinel value: pdays = -1 means "never contacted before"
SELECT SUM(CASE WHEN pdays = -1 THEN 1 ELSE 0 END)                  AS never_contacted_rows,
       ROUND(100.0 * SUM(CASE WHEN pdays = -1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS never_contacted_pct
FROM bank_marketing;


-- =====================================================================
-- SECTION 2: BASELINE
-- =====================================================================

-- 2.1 Class balance (window function over an aggregate gives the share)
SELECT y,
       COUNT(*) AS clients,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM bank_marketing
GROUP BY y;


-- =====================================================================
-- SECTION 3: CONVERSION ANALYSIS
-- =====================================================================

-- 3.1 Conversion by month with volume, ranks and a small-sample flag (CTE + RANK)
WITH monthly AS (
    SELECT month,
           COUNT(*) AS contacts,
           SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscribers
    FROM bank_marketing
    GROUP BY month
)
SELECT month,
       contacts,
       subscribers,
       ROUND(100.0 * subscribers / contacts, 1) AS conversion_pct,
       RANK() OVER (ORDER BY 1.0 * subscribers / contacts DESC) AS conversion_rank,
       RANK() OVER (ORDER BY contacts DESC)                      AS volume_rank,
       CASE WHEN contacts < 1000 THEN 'low volume: read with caution' ELSE 'ok' END AS sample_flag
FROM monthly
ORDER BY conversion_rank;


-- 3.2 Effort versus yield by month: where does the calling effort go?
WITH m AS (
    SELECT month,
           COUNT(*) AS c,
           SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS s
    FROM bank_marketing
    GROUP BY month
)
SELECT month,
       ROUND(100.0 * c / SUM(c) OVER (), 1) AS pct_of_all_contacts,
       ROUND(100.0 * s / SUM(s) OVER (), 1) AS pct_of_all_subscribers,
       ROUND(100.0 * s / c, 1)              AS conversion_pct
FROM m
ORDER BY pct_of_all_contacts DESC;


-- 3.3 Previous-campaign outcome: conversion and lift versus the overall rate
WITH overall AS (
    SELECT 1.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*) AS base_rate
    FROM bank_marketing
)
SELECT b.poutcome,
       COUNT(*) AS contacts,
       ROUND(100.0 * AVG(CASE WHEN b.y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct,
       ROUND(AVG(CASE WHEN b.y = 'yes' THEN 1.0 ELSE 0 END) / o.base_rate, 2) AS lift_vs_base
FROM bank_marketing b
CROSS JOIN overall o
GROUP BY b.poutcome, o.base_rate
ORDER BY lift_vs_base DESC;


-- 3.4 Contact fatigue: conversion by number of contacts in the campaign
SELECT CASE
           WHEN campaign = 1 THEN '1'
           WHEN campaign = 2 THEN '2'
           WHEN campaign = 3 THEN '3'
           WHEN campaign BETWEEN 4 AND 5 THEN '4-5'
           WHEN campaign BETWEEN 6 AND 10 THEN '6-10'
           ELSE '11+'
       END AS contacts_in_campaign,
       COUNT(*) AS clients,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct
FROM bank_marketing
GROUP BY 1
ORDER BY MIN(campaign);


-- 3.5 Age buckets: conversion by life stage (non-linear age effect)
SELECT CASE
           WHEN age <= 30 THEN '1_young (<=30)'
           WHEN age <= 45 THEN '2_middle (31-45)'
           WHEN age <= 60 THEN '3_senior (46-60)'
           ELSE '4_elderly (60+)'
       END AS age_group,
       COUNT(*) AS clients,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct
FROM bank_marketing
GROUP BY 1
ORDER BY 1;


-- 3.6 Occupation ranking by conversion (RANK window function)
SELECT job,
       COUNT(*) AS clients,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct,
       RANK() OVER (ORDER BY AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END) DESC) AS conversion_rank
FROM bank_marketing
GROUP BY job
ORDER BY conversion_rank;


-- 3.7 Debt status: conversion by loan combination
SELECT housing,
       loan,
       COUNT(*) AS clients,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct
FROM bank_marketing
GROUP BY housing, loan
ORDER BY conversion_pct DESC;


-- =====================================================================
-- SECTION 4: REUSABLE VIEWS
-- =====================================================================

-- 4.1 Feature view: the SQL equivalent of the Python feature engineering
CREATE OR REPLACE VIEW vw_bank_features AS
SELECT client_id,
       age, job, marital, education, credit_default, balance, housing, loan,
       contact, month, campaign, pdays, previous, poutcome, y,
       CASE WHEN pdays <> -1 THEN 1 ELSE 0 END AS was_previously_contacted,
       CASE WHEN age <= 30 THEN 'young'
            WHEN age <= 45 THEN 'middle'
            WHEN age <= 60 THEN 'senior'
            ELSE 'elderly' END AS age_group,
       CASE WHEN campaign <= 2 THEN 'low'
            WHEN campaign <= 5 THEN 'medium'
            ELSE 'high' END AS campaign_intensity,
       CASE WHEN housing = 'yes' OR loan = 'yes' THEN 1 ELSE 0 END AS has_any_loan
FROM bank_marketing;


-- 4.2 Monthly conversion view for reporting and dashboards
CREATE OR REPLACE VIEW vw_conversion_by_month AS
SELECT month,
       COUNT(*) AS contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscribers,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct
FROM bank_marketing
GROUP BY month;


-- 4.3 Using the feature view: conversion by campaign intensity and prior contact
SELECT campaign_intensity,
       was_previously_contacted,
       COUNT(*) AS clients,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END), 1) AS conversion_pct
FROM vw_bank_features
GROUP BY campaign_intensity, was_previously_contacted
ORDER BY campaign_intensity, was_previously_contacted;


-- 4.4 Using the monthly view: months above the overall conversion rate
SELECT month, contacts, conversion_pct
FROM vw_conversion_by_month
WHERE conversion_pct > (SELECT 100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0 END)
                        FROM bank_marketing)
ORDER BY conversion_pct DESC;


-- =====================================================================
-- SECTION 5: MODEL OUTPUT ANALYSIS (uses lead_scores)
-- =====================================================================

-- 5.1 Decile gain and lift table (NTILE splits scored leads into 10 equal groups)
WITH ranked AS (
    SELECT client_id, prob_yes, actual_yes,
           NTILE(10) OVER (ORDER BY prob_yes DESC) AS decile
    FROM lead_scores
),
decile_stats AS (
    SELECT decile,
           COUNT(*) AS clients,
           SUM(actual_yes) AS subscribers,
           ROUND(MIN(prob_yes), 3) AS min_score_in_decile
    FROM ranked
    GROUP BY decile
),
totals AS (
    SELECT SUM(actual_yes) AS total_subs,
           1.0 * SUM(actual_yes) / COUNT(*) AS base_rate
    FROM lead_scores
)
SELECT d.decile,
       d.clients,
       d.subscribers,
       d.min_score_in_decile,
       ROUND(100.0 * d.subscribers / d.clients, 1) AS precision_pct,
       ROUND(100.0 * SUM(d.subscribers) OVER (ORDER BY d.decile) / t.total_subs, 1) AS cumulative_capture_pct,
       ROUND((1.0 * d.subscribers / d.clients) / t.base_rate, 2) AS lift
FROM decile_stats d
CROSS JOIN totals t
ORDER BY d.decile;


-- 5.2 Headline business rule: call the top 20% (deciles 1 and 2)
WITH ranked AS (
    SELECT actual_yes,
           NTILE(10) OVER (ORDER BY prob_yes DESC) AS decile
    FROM lead_scores
)
SELECT ROUND(100.0 * SUM(CASE WHEN decile <= 2 THEN actual_yes ELSE 0 END)
             / SUM(actual_yes), 1) AS pct_of_subscribers_captured_in_top20,
       ROUND(100.0 * SUM(CASE WHEN decile <= 2 THEN actual_yes ELSE 0 END)
             / SUM(CASE WHEN decile <= 2 THEN 1 ELSE 0 END), 1) AS precision_in_top20_pct,
       ROUND(100.0 * SUM(actual_yes) / COUNT(*), 1) AS baseline_conversion_pct
FROM ranked;


-- 5.3 Segment check: average predicted score versus actual conversion by month
SELECT b.month,
       COUNT(*) AS test_clients,
       ROUND(100.0 * AVG(s.prob_yes), 1)   AS avg_predicted_pct,
       ROUND(100.0 * AVG(s.actual_yes), 1) AS actual_conversion_pct
FROM lead_scores s
JOIN bank_marketing b ON b.client_id = s.client_id
GROUP BY b.month
HAVING COUNT(*) >= 30
ORDER BY test_clients DESC;


-- 5.4 Missed opportunities: actual subscribers the model scored in the bottom half
WITH ranked AS (
    SELECT client_id, actual_yes,
           NTILE(10) OVER (ORDER BY prob_yes DESC) AS decile
    FROM lead_scores
)
SELECT b.poutcome,
       b.contact,
       COUNT(*) AS missed_subscribers
FROM ranked r
JOIN bank_marketing b ON b.client_id = r.client_id
WHERE r.actual_yes = 1 AND r.decile >= 6
GROUP BY b.poutcome, b.contact
ORDER BY missed_subscribers DESC;
SELECT COUNT(*) FROM lead_scores;     -- expect 9043

 