package patterns.command.timer;

import java.util.Date;

public class SchedulerTest {

	public static void main(String[] args) throws Exception {
		Scheduler scheduler = new SchedulerImpl();
		System.out.println("starting test at " + new Date());
		scheduler.register(new Printer("task1"), new Date(System.currentTimeMillis() +  5_000));
		scheduler.register(new Printer("task2"), new Date(System.currentTimeMillis() + 10_000));
		Thread.sleep( 1_000);
		scheduler.register(new Printer("task3"), new Date(System.currentTimeMillis() +  1_000));
		Thread.sleep(10_000);
	}
	
	static class Printer implements SchedulerTask {
		private volatile boolean cancel = false;
		private String msg;
		
		public Printer(String msg) {
			this.msg = msg;
		}

		@Override
		public void cancel() { cancel = true; }

		@Override
		public boolean isCancelled() { return cancel; }

		@Override
		public void execute() {
			System.out.println(msg + " at "+new Date());
		}
	}

}
