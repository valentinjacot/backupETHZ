package Villes;

public class Ville {
	  
	  /**
	   * Variable publique compteur d'instances
	   */
	  public static int nbreInstance = 0;
	  /**
	   * Variable privée compteur d'instances
	   */
	  protected static int nbreInstanceBis = 0;
	         
	  /**
	   * Stocke le nom de notre ville
	   */
	  protected String nomVille;
	  /**
	   * Stocke le nom du pays de notre ville
	   */
	  protected String nomPays;
	  /**
	   * Stocke le nombre d'habitants de notre ville
	   */
	  protected int nbreHabitant;
	  /**
	   * Stocke le type de notre ville
	   */
	  protected char categorie;
	   
	  /**
	   * Constructeur par défaut
	   */
	  public Ville(){
	          //On incrémente nos variables à chaque appel au constructeur
	          nbreInstance++;
	          nbreInstanceBis++;
	           
	          nomVille = "Inconnu";
	          nomPays = "Inconnu";
	          nbreHabitant = 0;
	          this.setCategorie();
	  }
	  
	  /**
	   * Constructeur d'initialisation
	   * @param pNom
	   *                    Nom de la Ville
	   *  @param pNbre
	   *                    Nombre d'habitants
	   *  @param pPays
	   *                    Nom du pays
	 * @throws 
	   */
	  public Ville(String pNom, int pNbre, String pPays) throws NombreHabitantException
	  { 
	      if(pNbre < 0)
	          throw new NombreHabitantException(pNbre);
	      else
	      {
	 
	          nbreInstance++;
	          nbreInstanceBis++;
	           
	          nomVille = pNom;
	          nomPays = pPays;
	          nbreHabitant = pNbre;
	          this.setCategorie();
	      }
	  }
	         
	   
	  //*****************************************************************************************
	  //                                    ACCESSEURS
	  //*****************************************************************************************
	   
	  public static int getNombreInstanceBis()
	  {
	          return nbreInstanceBis;
	  }
	   
	  /**
	   * Retourne le nom de la ville
	   * @return le nom de la ville
	   */
	  public String getNom()
	  {
	          return nomVille;
	  }
	   
	  /**
	   * Retourne le nom du pays
	   * @return le nom du pays
	   */
	  public String getNomPays()
	  {
	          return nomPays;
	  }
	   
	  /**
	   * Retourne le nombre d'habitants
	   * @return nombre d'habitants
	   */
	 public int getNombreHabitant()
	 {
	         return nbreHabitant;
	 }
	  
	 /**
	  * Retourne la catégorie de la ville
	  * @return catégorie de la ville 
	  */
	 public char getCategorie()
	 {
	         return categorie;
	 }
	  
	 //*****************************************************************************************
	 //                                    MUTATEURS
	 //*****************************************************************************************
	  
	 /**
	  * Définit le nom de la ville
	  * @param pNom
	  *             nom de la ville
	  */
	 public void setNom(String pNom)
	 {
	          nomVille = pNom;
	 }
	  
	 /**
	  * Définit le nom du pays
	  * @param pPays
	  *             nom du pays
	  */
	 public void setNomPays(String pPays)
	 {
	          nomPays = pPays;
	 }
	  
	 /**
	  * Définit le nombre d'habitants
	  * @param nbre
	  *             nombre d'habitants
	  */
	public void setNombreHabitant(int nbre)
	{
	         nbreHabitant = nbre;
	         this.setCategorie();
	}
	  
	  
	  
	//*****************************************************************************************
//	                                    METHODES DE CLASSE
	//*****************************************************************************************
	  
	  
	  
	  /**
	   * Définit la catégorie de la ville
	   */
	  protected void setCategorie() {
	  
	       
	      int bornesSuperieures[] = {0, 1000, 10000, 100000, 500000, 1000000, 5000000, 10000000};
	        char categories[] = {'?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'};
	 
	        int i = 0;
	        while (i < bornesSuperieures.length && this.nbreHabitant >= bornesSuperieures[i])
	                i++;
	 
	        this.categorie = categories[i];
	 
	  
	  }
	  
	  /**
	   * Retourne la description de la ville
	   * @return description ville
	   */
	  public String toString(){
	      return "\t"+this.nomVille+" est une ville de "+this.nomPays+", elle comporte : "+this.nbreHabitant+
	              " => elle est donc de catégorie : "+this.categorie;
	  }
	  
	  /**
	   * Retourne une chaîne de caractères selon le résultat de la comparaison
	   * @param v1
	   *            objet Ville
	   * @return comparaison de deux ville
	   */
	  public String comparer(Ville v1){
	      String str = new String();
	      
	      if (v1.getNombreHabitant() > this.nbreHabitant)
	          str = v1.getNom()+" est une ville plus peuplée que "+this.nomVille;
	      
	      else
	          str = this.nomVille+" est une ville plus peuplée que "+v1.getNom();
	      
	      return str;
	      
	  }
	  
	}