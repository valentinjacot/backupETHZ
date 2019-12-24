package Button;

import java.awt.BorderLayout;

import javax.swing.JButton;
import javax.swing.JFrame;

public class FenetreBorderLayout extends JFrame{
//	private JPanel pan = new JPanel();
//	private JButton bouton = new JButton("Mon bouton");
	
	public FenetreBorderLayout () {
		this.setTitle("Animation");
		this.setSize(300, 300);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setLocationRelativeTo(null);
		this.setLayout(new BorderLayout());
		this.getContentPane().add(new JButton("Center"),BorderLayout.CENTER);
		this.getContentPane().add(new JButton("South"),BorderLayout.SOUTH);
		this.getContentPane().add(new JButton("North"),BorderLayout.NORTH);
		this.getContentPane().add(new JButton("West"),BorderLayout.WEST);
		this.getContentPane().add(new JButton("East"),BorderLayout.EAST);
		this.setVisible(true);
		
	}
}
