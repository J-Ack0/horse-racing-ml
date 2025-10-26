-- triggers and test for movie table
USE cinema;

-- 1) Audit table creation
-- simple create table for the audit table, similar vars to movie but with log_id, action_type, action_time (for audit purposes) 
-- and old / new variants of prior variables

DROP TABLE IF EXISTS movie_audit;
CREATE TABLE movie_audit (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    action_type VARCHAR(20),
    movie_id INT,
    new_box_office DECIMAL(12,2),
    old_box_office DECIMAL(12,2),
    old_title VARCHAR(150),
    new_title VARCHAR(150),
    action_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2) Triggers
-- Trigger SQL, Insert, update and delete
DELIMITER $$

-- TRIGGER Functionalty == on insert show some vars.
CREATE TRIGGER trigger_movie_after_insert
AFTER INSERT ON movie
FOR EACH ROW
BEGIN
    INSERT INTO movie_audit (action_type, movie_id, new_title, new_box_office, old_box_office)
    VALUES ('INSERT', NEW.movie_id, NEW.title, NEW.box_office, NULL);
END$$

--TRIGGER == on update show changes: 
CREATE TRIGGER trigger_movie_after_update
AFTER UPDATE ON movie
FOR EACH ROW
BEGIN
    INSERT INTO movie_audit (action_type, movie_id, old_title, new_title, new_box_office, old_box_office)
    VALUES ('UPDATE', NEW.movie_id, OLD.title, NEW.title, NEW.box_office, OLD.box_office);
END$$

--TRIGGER functionalty == on delete show deleted item.
CREATE TRIGGER trigger_movie_after_delete
AFTER DELETE ON movie
FOR EACH ROW
BEGIN
    INSERT INTO movie_audit (action_type, movie_id, old_title, old_box_office, new_title, new_box_office)
    VALUES ('DELETE', OLD.movie_id, OLD.title, OLD.box_office, NULL, NULL);
END$$

DELIMITER ;

-- 3) Tests
-- Simple tests to prove working triggers, insert update and delete, Box office example for update as its most likley to chnage over time.
-- INSERT TRIGGER TEST
INSERT INTO movie (title, release_date, genre, duration_in_minutes, language, budget, box_office, studio_id)
VALUES ('Test Movie', '2025-10-23', 'Horror', 120, 'English', 20000000.00, 85000000.00, 1);

-- UPDATE TRIGGER TEST
UPDATE movie
SET box_office = 90000000.00
WHERE box_office = 85000000.00;

-- DELETE TRIGGER TEST
DELETE FROM movie
WHERE title = 'Test Movie';

-- 4) Verify
-- Shows AUDIT TABLE for movie Insert,delete, update self explanatory when ran. 
SELECT * FROM movie_audit ORDER BY log_id;
