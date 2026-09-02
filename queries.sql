-- Analytical SQL Queries for MGC CRM Leads


-- Query 1: Conversion rate by lead source
-- Filter for channels with at least 200 total leads and rank by conversion rate descending.

SELECT
    source,
    COUNT(*)                                    AS total_leads,
    SUM(converted)                              AS conversions,
    ROUND(AVG(converted) * 100, 2)              AS conversion_rate_pct
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;


-- Query 2: Find duplicate lead entries
-- Identifies leads entered multiple times by different agents using the unique identity hash.

SELECT
    a.lead_id       AS lead_id_1,
    b.lead_id       AS lead_id_2,
    a.crm_record_hash,
    a.source        AS source_1,
    b.source        AS source_2,
    a.city          AS city_1,
    b.city          AS city_2,
    a.created_at    AS created_at_1,
    b.created_at    AS created_at_2
FROM leads a
JOIN leads b
    ON  a.crm_record_hash = b.crm_record_hash
    AND a.lead_id < b.lead_id       -- prevents matching a row with itself and removes inverse pairs
ORDER BY a.crm_record_hash;

-- Duplicate Prevention Note:
-- The `UNIQUE(crm_record_hash)` constraint in schema.sql blocks duplicate inserts automatically.
-- Trying to insert an existing hash will raise a constraint failure:
--
--     INSERT INTO leads (..., crm_record_hash, ...) VALUES (..., 1007444249, ...);
--     --> UNIQUE constraint failed: leads.crm_record_hash
--
-- To handle updates gracefully on conflict:
--     INSERT OR REPLACE INTO leads (...) VALUES (...);         -- SQLite
--     INSERT INTO leads (...) VALUES (...) ON CONFLICT (...)   -- PostgreSQL
