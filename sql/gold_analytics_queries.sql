-- Gold analytical queries for the local NYC Taxi Delta Lake.

CREATE OR REPLACE TEMP VIEW gold_trip_metrics
USING delta
OPTIONS (
    path 'data/lakehouse/gold/trip_metrics_by_date_pickup_zone_payment'
);

CREATE OR REPLACE TEMP VIEW gold_daily_metrics
USING delta
OPTIONS (
    path 'data/lakehouse/gold/daily_trip_metrics'
);


-- 1. Daily accepted-trip activity and revenue.

SELECT
    pickup_date,
    trip_count,
    quality_flagged_trip_count,
    ROUND(total_amount_sum, 2) AS total_amount_sum,
    ROUND(total_amount_avg, 2) AS total_amount_avg
FROM gold_daily_metrics
ORDER BY pickup_date;


-- 2. Top pickup zones by accepted-trip volume.

SELECT
    pickup_borough,
    pickup_zone,
    SUM(trip_count) AS accepted_trip_count,
    ROUND(SUM(total_amount_sum), 2) AS total_amount_sum,
    ROUND(SUM(tip_amount_sum), 2) AS tip_amount_sum
FROM gold_trip_metrics
GROUP BY
    pickup_borough,
    pickup_zone
ORDER BY accepted_trip_count DESC
LIMIT 20;


-- 3. Accepted-trip metrics by TLC payment-type code.

SELECT
    payment_type,
    SUM(trip_count) AS accepted_trip_count,
    SUM(quality_flagged_trip_count) AS quality_flagged_trip_count,
    ROUND(SUM(total_amount_sum), 2) AS total_amount_sum,
    ROUND(SUM(tip_amount_sum), 2) AS tip_amount_sum
FROM gold_trip_metrics
GROUP BY payment_type
ORDER BY accepted_trip_count DESC;


-- 4. Monthly retained-quality-flag rate.

SELECT
    _source_month,
    SUM(trip_count) AS accepted_trip_count,
    SUM(quality_flagged_trip_count) AS quality_flagged_trip_count,
    ROUND(
        100.0
        * SUM(quality_flagged_trip_count)
        / SUM(trip_count),
        2
    ) AS quality_flagged_trip_percent
FROM gold_daily_metrics
GROUP BY _source_month
ORDER BY _source_month;


-- 5. Gold row-preservation validation.

SELECT
    'trip_metrics' AS dataset,
    COUNT(*) AS aggregate_row_count,
    SUM(trip_count) AS accepted_trip_count,
    SUM(quality_flagged_trip_count) AS quality_flagged_trip_count
FROM gold_trip_metrics

UNION ALL

SELECT
    'daily_metrics' AS dataset,
    COUNT(*) AS aggregate_row_count,
    SUM(trip_count) AS accepted_trip_count,
    SUM(quality_flagged_trip_count) AS quality_flagged_trip_count
FROM gold_daily_metrics;


-- 6. Gold grain-uniqueness validation.

SELECT
    'trip_metrics' AS dataset,
    COUNT(*) AS duplicate_grain_count
FROM (
    SELECT
        _source_month,
        pickup_date,
        pickup_location_id,
        payment_type
    FROM gold_trip_metrics
    GROUP BY
        _source_month,
        pickup_date,
        pickup_location_id,
        payment_type
    HAVING COUNT(*) > 1
)

UNION ALL

SELECT
    'daily_metrics' AS dataset,
    COUNT(*) AS duplicate_grain_count
FROM (
    SELECT
        _source_month,
        pickup_date
    FROM gold_daily_metrics
    GROUP BY
        _source_month,
        pickup_date
    HAVING COUNT(*) > 1
);
