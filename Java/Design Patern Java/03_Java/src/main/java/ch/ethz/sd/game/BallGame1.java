package ch.ethz.sd.game;


import javax.swing.*;
import java.awt.event.*;


@SuppressWarnings("serial")
public class BallGame1 extends JFrame {
	public static void main(String[] args){
		JFrame f = new BallGame1();
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}
	
	public BallGame1(){
		setTitle("BallGame1");
		add(new BallField1());
	}
}

@SuppressWarnings("serial")
class BallField1 extends JComponent  implements MouseListener {
	// position and radius of a ball
	private int x = 50, y = 50, r = 10;
	
	public BallField1(){
		setPreferredSize(new java.awt.Dimension(300,300));
		addMouseListener(this);
	}

	@Override
	public void paintComponent (java.awt.Graphics g) {
		g.setColor(java.awt.Color.red);
		g.fillOval(x-r, y-r, 2*r, 2*r);
	}
  
	public void mouseClicked(MouseEvent e){}
	public void mouseEntered(MouseEvent e){}
	public void mouseExited (MouseEvent e){}
	public void mousePressed(MouseEvent e){
		x = e.getX();
		y = e.getY();
		repaint();
	}
	public void mouseReleased(MouseEvent e){}
}
