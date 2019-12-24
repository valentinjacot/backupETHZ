package ch.ethz.sd.game;


import javax.swing.*;
import java.awt.event.*;


@SuppressWarnings("serial")
public class BallGame2 extends JFrame {
	public static void main(String[] args){
		JFrame f = new BallGame2();
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}
	
	public BallGame2(){
		setTitle("BallGame2");
		add(new BallField2());
	}
}

@SuppressWarnings("serial")
class BallField2 extends JComponent {
	// position and radius of a ball
	int x = 50, y = 50, r = 10;
	
	public BallField2(){
		setPreferredSize(new java.awt.Dimension(300,300));
		addMouseListener(new BallFieldMouseListener2(this));
	}

	public void paintComponent (java.awt.Graphics g) {
		g.setColor(java.awt.Color.red);
		g.fillOval(x-r, y-r, 2*r, 2*r);
	}
}

class BallFieldMouseListener2 implements MouseListener {
	private BallField2 field;
	BallFieldMouseListener2(BallField2 field){
		this.field = field;
	}
	
	public void mouseClicked(MouseEvent e){}
	public void mouseEntered(MouseEvent e){}
	public void mouseExited (MouseEvent e){}
	public void mouseReleased(MouseEvent e){}
	public void mousePressed(MouseEvent e){
		field.x = e.getX();
		field.y = e.getY();
		field.repaint();
	}
}

	