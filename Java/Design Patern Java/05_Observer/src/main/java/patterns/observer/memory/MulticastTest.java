package patterns.observer.memory;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public class MulticastTest {

	public static void main(String[] args) {
		SwingUtilities.invokeLater(() -> {
			JFrame frame = new JFrame() {
				{
					this.setTitle("MulticastTest");
					this.setSize(300, 75);
					this.add(new MulticastPanel());
				}
			};
	
			frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
			frame.setVisible(true);
		});
	}

}
