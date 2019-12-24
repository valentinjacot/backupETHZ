package ch.ethz.sd.app;
import java.awt.FlowLayout;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public class SimpleAppSwing {

	public static void main(String[] args) {
		SwingUtilities.invokeLater(() -> {
			JFrame f = new SimpleFrame();
			f.pack();
			f.setVisible(true);
		});
	}
	
	private static class SimpleFrame extends JFrame {
	
		private SimpleFrame() {
			super("Simple Application");
			
			JTextField text = new JTextField(20);
			JButton button = new JButton("Submit");

			setLayout(new FlowLayout());
			add(text);
			add(button);
			button.addActionListener(
				e -> System.out.println("[Swing] Submit: " + text.getText())
			);
	
			JMenuBar mbar = new JMenuBar();
			setJMenuBar(mbar);
	
			JMenu m = new JMenu("File");
			mbar.add(m);
	
			m.add("New");
			m.addSeparator();
			JMenuItem exit = new JMenuItem("Exit");
			exit.addActionListener(e -> System.exit(0));
			m.add(exit);
	
			setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		}
	
	}

}