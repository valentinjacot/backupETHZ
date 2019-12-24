package Villes;

public class Sdz1 {
	  
    public static void main(String[] args)
    {
            Ville v = null;
            try {                  
                    v = new Ville("Rennes", -12000, "France");              
            } catch (NombreHabitantException e) {}
            finally {
            	if(v==null)
            		v=new Ville();
            }
            System.out.println(v.toString());
    }      
       
}