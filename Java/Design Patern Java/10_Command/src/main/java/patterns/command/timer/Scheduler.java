package patterns.command.timer;

import java.util.Date;

public interface Scheduler {
	void register(SchedulerTask task, Date at);
	void purgeAllTasks();
}
