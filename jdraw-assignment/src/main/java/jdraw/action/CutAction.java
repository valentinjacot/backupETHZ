package jdraw.action;

import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.KeyStroke;
import javax.swing.event.MenuEvent;

import jdraw.framework.DrawContext;
import jdraw.framework.Figure;
import jdraw.std.SimpleClipboard;

@SuppressWarnings("serial")
public class CutAction extends AbstractAction{
	public CutAction(DrawContext dc, JMenu JM) {
		super(dc);
		putValue(Action.SHORT_DESCRIPTION, "cuts all selected figures");
		putValue(Action.NAME, "cut");
        putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("CTRL-X"));
		JM.addMenuListener(this);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		List<Figure> selection = actionContext.getView().getSelection();
		if(selection != null) {
			SimpleClipboard.clear();
			for (Figure f : selection) {
				SimpleClipboard.add(f.clone());
				actionContext.getModel().removeFigure(f);
			}
		}
	}

	
	@Override
	public void menuSelected(MenuEvent e) {
		setEnabled(actionContext.getView().getSelection().size() > 0);
	}

	
}
