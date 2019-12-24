package ch.ethz.sd.app;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;

import javax.swing.JButton;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

@SuppressWarnings("serial")
class SwingApp extends JFrame {
	
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				JFrame f = new SwingApp();
				f.pack();
				f.setVisible(true);
			}
		});
	}

	SwingApp() {
		super("Swing Application");
		setLayout(new BorderLayout());

		JTextField text1 = new JTextField();
		JButton button1 = new JButton("Button1");
		JButton button2 = new JButton("Button2");
		JButton button3 = new JButton("Button3");
		JButton button4 = new JButton("Button4");
		JButton button5 = new JButton("Button5");
		JButton button6 = new JButton("Button6");

		JPanel center = new JPanel();
		center.setLayout(new FlowLayout());

		JPanel centerleft = new JPanel();
		centerleft.setLayout(new GridLayout(3, 2));
		centerleft.add(button1);
		centerleft.add(button2);
		centerleft.add(button3);
		centerleft.add(button4);
		centerleft.add(button5);

		JPanel centerright = new JPanel();
		centerright.setLayout(new GridLayout(0, 1));
		centerright.add(new JRadioButton("Option 1"));
		centerright.add(new JRadioButton("Option 2"));
		centerright.add(new JRadioButton("Option 3"));
		centerright.add(new JRadioButton("Option 4"));

		center.add(centerleft);
		center.add(centerright);

		add(text1, BorderLayout.NORTH);
		add(center, BorderLayout.CENTER);
		add(button6, BorderLayout.SOUTH);

		JCheckBoxMenuItem item1 = new JCheckBoxMenuItem("Check", true);
		JRadioButtonMenuItem item2 = new JRadioButtonMenuItem("Radio", true);
		JMenuItem item3 = new JMenuItem("Exit");

		item3.addActionListener(e -> System.exit(0));

		JMenuBar mb = new JMenuBar();
		setJMenuBar(mb);

		JMenu file = new JMenu("File");
		mb.add(file);

		file.add("Open...");
		file.addSeparator();
		file.add(item1);
		file.add(item2);
		file.add(item3);

		// enables the window's X (close) button
		this.setDefaultCloseOperation(EXIT_ON_CLOSE); 
	}
}
