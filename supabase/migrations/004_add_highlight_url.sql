-- Add highlight_url column to pbp for goal video links
ALTER TABLE pbp ADD COLUMN IF NOT EXISTS highlight_url TEXT;
