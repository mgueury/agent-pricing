create table gc_products (
    product_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_name VARCHAR2(100) NOT NULL,
    product_type VARCHAR2(50)  NOT NULL,
    DESCRIPTION      VARCHAR2(500),
    unit_cost NUMBER(12, 2) NOT NULL,
    currency         VARCHAR2(3) DEFAULT 'EUR'
);
BEGIN
    -- Transcoding Services
    INSERT INTO gc_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('Transcode 25Mbps 1080p', 'transcode', 'High bitrate HD transcode per hour.', 100, 'EUR');

    INSERT INTO gc_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('Transcode 15Mbps 720p', 'transcode', 'Mid bitrate HD transcode per hour.', 70, 'EUR');

    INSERT INTO gc_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('Transcode 5Mbps 480p', 'transcode', 'Low bitrate SD transcode per hour.', 40, 'EUR');

    -- Distribution Services
    INSERT INTO gc_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('CDN Distribution - Global', 'distribution', 'CDN-based distribution to global regions per CDN per hour.', 100, 'EUR');

    INSERT INTO gc_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('Direct IP Distribution', 'distribution', 'IP-based direct distribution service per taker per hours.', 5, 'EUR');

    -- Redundancy & Disaster Recovery
    --INSERT INTO gc_products (product_name, product_type, description, unit_cost, currency)
    --VALUES ('Redundancy Service', 'redundancy', 'Disaster recovery standby feed per hour.', 150, 'USD');
    --COMMIT;
END;
/
create table oci_products (
    product_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_name VARCHAR2(100) NOT NULL,
    product_type VARCHAR2(50)  NOT NULL,
    DESCRIPTION      VARCHAR2(500),
    unit_cost NUMBER(12, 5) NOT NULL,
    currency         VARCHAR2(3) DEFAULT 'EUR'
);
BEGIN
    INSERT INTO oci_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('Compute E5 - OCPU', 'CPU', 'Cost of 1 OCPU per hour', 0.0279, 'EUR');

    INSERT INTO oci_products (product_name, product_type, description, unit_cost, currency)
    VALUES ('Compute E5 - Memory', 'Memory', 'Cost of 1 GB of Memory/RAM per hour', 0.00186, 'EUR');
END;
/