package classesAbstraites;

abstract class Animal {
  
        /**
         * La couleur de l'animal
         */
        protected String couleur;
        /**
         * Le poids
         */
        protected int poids;
         
        /**
         * La méthode manger
         */
        protected void manger(){
                System.out.println("Je mange de la viande");
        }
         
        /**
         * La méthode boire
         */
        protected void boire(){
                System.out.println("Je bois de l'eau !");
        }
         
        /**
         * La méthode de déplacement
         */
        abstract void deplacement();
        /**
         * La méthode de cri
         */
        abstract void crier();
         
        public String toString(){
                 
                String str = "Je suis un objet de la " + this.getClass() + ", je suis " + this.couleur + ", je pèse " + this.poids;
                return str;
        }
         
}