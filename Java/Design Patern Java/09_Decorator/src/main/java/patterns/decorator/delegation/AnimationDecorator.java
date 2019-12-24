package patterns.decorator.delegation;

public class AnimationDecorator extends AbstractDecorator {

	public AnimationDecorator(Figure inner) {
		super(inner);
		Thread t = new Thread(() -> {
			while (true) {
				try {
					Thread.sleep(2000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				move(2, 2);
				draw();
			}
		});
		t.start();
	}

}
