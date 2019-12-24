package patterns.command.timer;

public interface SchedulerTask {
	void execute();
	void cancel();
	boolean isCancelled();
}
