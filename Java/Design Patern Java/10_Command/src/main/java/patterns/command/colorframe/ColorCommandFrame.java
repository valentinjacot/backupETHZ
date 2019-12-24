package patterns.command.colorframe;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public class ColorCommandFrame extends JFrame {

	public static void main(String[] args) {
		JFrame f = new ColorCommandFrame();
		f.setSize(400, 300);
		f.setVisible(true);
	}

	private final ImageIcon yellowIcon = new ImageIcon(this.getClass().getResource("/yellow_bullet.gif"));
	private final ImageIcon redIcon    = new ImageIcon(this.getClass().getResource("/red_bullet.gif"));

	private final JPanel coloredPanel = new JPanel();

	public ColorCommandFrame() {
		super("ColorCommandFrame");
		setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		initButtons();
		initPopupMenu();
		initMenu();
	}

	public void initButtons() {
		JButton yellowButton = new JButton("yellow", yellowIcon);
		JButton redButton = new JButton("red", redIcon);
		yellowButton.addActionListener(e -> yellowAction(e));
		redButton.addActionListener(this::redAction);
		coloredPanel.add(yellowButton);
		coloredPanel.add(redButton);
		add(coloredPanel);
	}

	public void initPopupMenu() {
		JMenuItem yellowPopupItem = new JMenuItem("yellow", yellowIcon);
		JMenuItem redPopupItem = new JMenuItem("red", redIcon);
		final JPopupMenu popupMenu = new JPopupMenu();
		yellowPopupItem.addActionListener(e -> yellowAction(e));
		redPopupItem.addActionListener(this::redAction);
		popupMenu.add(yellowPopupItem);
		popupMenu.add(redPopupItem);

		coloredPanel.addMouseListener(new MouseAdapter() {
			public @Override
			void mousePressed(MouseEvent e) {
				maybeShowPopup(e);
			}

			public @Override
			void mouseReleased(MouseEvent e) {
				maybeShowPopup(e);
			}

			private void maybeShowPopup(MouseEvent e) {
				if (e.isPopupTrigger()) {
					popupMenu.show(e.getComponent(), e.getX(), e.getY());
				}
			}
		});
	}

	public void initMenu() {
		JMenuBar menuBar = new JMenuBar();
		JMenu colorMenu = new JMenu("Color");
		colorMenu.setMnemonic(KeyEvent.VK_C);
		JMenuItem yellowMenuItem = new JMenuItem("yellow", yellowIcon);
		JMenuItem redMenuItem = new JMenuItem("red", redIcon);
		yellowMenuItem.addActionListener(e -> yellowAction(e));
		redMenuItem.addActionListener(this::redAction);
		colorMenu.add(yellowMenuItem);
		colorMenu.add(redMenuItem);
		menuBar.add(colorMenu);
		setJMenuBar(menuBar);
	}

	public void redAction(ActionEvent e) {
		coloredPanel.setBackground(Color.RED);
	}
	
	public void yellowAction(ActionEvent e) {
		coloredPanel.setBackground(Color.YELLOW);
	}
}
