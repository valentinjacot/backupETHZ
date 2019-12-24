package patterns.strategy.stateful;

import java.awt.GridLayout;
import java.awt.LayoutManager;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

@SuppressWarnings("serial")
public class LayoutComparer extends JFrame {

	public static void main(String[] args) {
		JFrame f = new LayoutComparer();
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}

	static int counter = 0;

	JPanel createPanel(LayoutManager layout, String title) {
		JPanel p = new JPanel();
		p.setLayout(layout);
		p.add(new JButton("Click " + counter++), "West");
		p.add(new JButton("Click " + counter++), "Center");
		p.add(new JButton("Click " + counter++), "East");
		p.setBorder(BorderFactory.createTitledBorder(title));
		return p;
	}

	LayoutComparer() {
		setTitle("Layout Manager Test");
		setLayout(new GridLayout(1, 2));
		LayoutManager m;
		m = new java.awt.FlowLayout();
//		m = new java.awt.BorderLayout();

		add(createPanel(m, "Left"));
//		pack();
		add(createPanel(m, "Right"));
	}
}