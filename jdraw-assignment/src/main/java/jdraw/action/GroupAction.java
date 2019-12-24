package jdraw.action;

import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.Action;
import javax.swing.ImageIcon;
import javax.swing.JMenu;
import javax.swing.KeyStroke;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import jdraw.commands.GroupCommand;
import jdraw.figures.Group;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawModel;
import jdraw.framework.Figure;


@SuppressWarnings("serial")
public class GroupAction extends AbstractAction implements MenuListener{
	
	public GroupAction (DrawContext AC, JMenu JM) {
		super(AC);		
		putValue(Action.SMALL_ICON, new ImageIcon(getClass().getResource(IMAGES + "group.png")));
		putValue(Action.SHORT_DESCRIPTION, "groups all selected figures");
		putValue(Action.NAME, "group");
        putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("CTRL-g"));
		JM.addMenuListener(this);
	}
	
	
	@Override
	public void menuSelected(MenuEvent ignore) {
		setEnabled(actionContext.getView().getSelection().size() > 1);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		List<Figure> selection = actionContext.getView().getSelection();
		if(selection != null && selection.size() >= 2 ) {
			Group g = new Group(selection);
			DrawModel dm = actionContext.getView().getModel();
			dm.getDrawCommandHandler().addCommand(
                    new GroupCommand(g, actionContext.getModel(), true)
					);
			for(Figure f : selection ) {
				dm.removeFigure(f);
				actionContext.getView().removeFromSelection(f);
			}
			dm.addFigure(g);
			actionContext.getView().addToSelection(g);
		}
		
	}

}
