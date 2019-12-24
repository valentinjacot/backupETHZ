package ch.ethz.sd.app;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JTextField;

@SuppressWarnings("serial")
public class SimpleAppAWT extends JFrame implements ActionListener {

	public static void main(String[] args) {
		JFrame f = new SimpleAppAWT();
		f.pack();
		f.setVisible(true);
	}

	private final JTextField text = new JTextField(20);

	private SimpleAppAWT() {
		super("Simple Application"); // set title

		JButton button = new JButton("Submit");

		setLayout(new FlowLayout());
		add(text);
		add(button);
		button.addActionListener(this);

		JMenuBar mbar = new JMenuBar();
		setJMenuBar(mbar);

		JMenu m = new JMenu("File");
		mbar.add(m);

		m.add("New");
		m.addSeparator();
		m.add(new JMenuItem("Exit"));
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
	}

	public void actionPerformed(ActionEvent e) {
		System.out.println("[AWT] Submit: " + text.getText());
	}

	static class CloseListener extends WindowAdapter {
		public void windowClosing(WindowEvent e) {
			System.exit(0);
		}
	}

}