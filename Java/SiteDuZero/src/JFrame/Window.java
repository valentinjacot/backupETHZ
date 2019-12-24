package JFrame;

import javax.swing.JFrame;

public class Window extends JFrame {
	public Window() {
		this.setTitle("Mon titre");
		this.setSize(200,250);
		this.setLocationRelativeTo(null);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setContentPane(new arcEnCiel());		
//		JPanel pan = new JPanel();
//		pan.setBackground(Color.ORANGE);
//		this.setContentPane(pan);
		this.setVisible(true);
	}
}
