package patterns.command.timer;

import java.util.Date;
import java.util.SortedMap;
import java.util.TreeMap;

public class SchedulerImpl implements Scheduler {

	private final SortedMap<Date, SchedulerTask> tasks = new TreeMap<>();

	@Override
	public void purgeAllTasks() {
		synchronized (tasks) {
			tasks.clear();
			tasks.notify();
		}
	}

	@Override
	public void register(SchedulerTask task, Date at) {
		synchronized (tasks) {
			tasks.put(at, task);
			tasks.notify();
		}
	}

	public SchedulerImpl() {
		SchedulerThread t = new SchedulerThread();
		t.setDaemon(true);
		t.start();
	}

	private class SchedulerThread extends Thread {
		@Override
		public void run() {
			while (true) {
				synchronized (tasks) {
					if (tasks.isEmpty()) {
						try {
							tasks.wait();
						} catch (InterruptedException e) {
							// ignore exception
						}
					} else {
						long w = tasks.firstKey().getTime() - System.currentTimeMillis();
						if (w <= 0) {
							SchedulerTask task = tasks.remove(tasks.firstKey());
							if (!task.isCancelled())
								task.execute();
						} else {
							try {
								tasks.wait(w);
							} catch (InterruptedException e) {
								// ignore exception
							}
						}
					}
				}
			}
		}
	}
}
