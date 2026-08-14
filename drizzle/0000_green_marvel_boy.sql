CREATE TABLE `daily_care_metrics` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`device_id` text NOT NULL,
	`date` text NOT NULL,
	`safety_alerts` integer DEFAULT 0 NOT NULL,
	`double_checks` integer DEFAULT 0 NOT NULL,
	`tasks_started` integer DEFAULT 0 NOT NULL,
	`tasks_completed` integer DEFAULT 0 NOT NULL,
	`micro_delay_samples` integer DEFAULT 0 NOT NULL,
	`micro_delay_slow_samples` integer DEFAULT 0 NOT NULL,
	`busy_level` text DEFAULT 'normal' NOT NULL,
	`synced_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
