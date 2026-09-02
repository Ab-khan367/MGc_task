-- SQL Schema for MGC CRM Leads Table (SQLite compatible)

-- Single table layout:
-- Since this is an export from a flat CRM dataset with ~9k rows, keeping it in one table
-- avoids unnecessary join overhead when querying. In a larger production app with concurrent
-- writes, I'd break out `cities` and `sources` into foreign key lookup tables.

CREATE TABLE IF NOT EXISTS leads (
    lead_id                     TEXT        PRIMARY KEY,            -- e.g. 'MGC-104067'
    created_at                  TIMESTAMP   NOT NULL,
    source                      TEXT        NOT NULL,               -- channel e.g. 'Facebook Ads', 'Referral'
    city                        TEXT        NOT NULL,               -- cleaned city name e.g. 'Islamabad'
    area                        TEXT,                               -- optional sub-locality
    property_type               TEXT        NOT NULL,               -- type e.g. 'Apartment', 'Villa'
    budget_pkr_lac              REAL,                               -- lead budget in Lakhs PKR
    bedrooms                    INTEGER,                            -- count of requested bedrooms
    first_response_minutes      REAL,                               -- minutes elapsed before first response
    calls_made                  INTEGER     NOT NULL DEFAULT 0,
    total_call_seconds          REAL        NOT NULL DEFAULT 0,
    whatsapp_replies            INTEGER     NOT NULL DEFAULT 0,
    site_visits                 INTEGER     NOT NULL DEFAULT 0,
    agent_experience_years      REAL,                               -- sales agent experience in years
    is_overseas                 BOOLEAN     NOT NULL DEFAULT 0,
    referred_by_existing_client BOOLEAN     NOT NULL DEFAULT 0,
    has_financing_approved      BOOLEAN     NOT NULL DEFAULT 0,
    token_amount_received_pkr   REAL        NOT NULL DEFAULT 0,
    crm_record_hash             BIGINT      NOT NULL UNIQUE,        -- unique hash prevents double-entry of leads
    converted                   BOOLEAN     NOT NULL DEFAULT 0
);

-- Index on lead source for filtering and aggregation speed
CREATE INDEX IF NOT EXISTS idx_leads_source ON leads(source);

-- Index on record hash to speed up deduplication lookups
CREATE INDEX IF NOT EXISTS idx_leads_hash ON leads(crm_record_hash);
