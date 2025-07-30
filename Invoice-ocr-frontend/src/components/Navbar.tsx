import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { FileText, LayoutDashboard, Users, UserCheck } from "lucide-react";

const Navbar = () => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="border-b bg-card/50 backdrop-blur-md border-border/50 sticky top-0 z-50">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-20">
          <div className="flex items-center space-x-12">
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="p-2 rounded-xl gradient-primary shadow-primary group-hover:shadow-glow transition-all duration-300">
                <FileText className="w-7 h-7 text-white" />
              </div>
              <span className="font-display font-bold text-2xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                OCR Invoice
              </span>
            </Link>
            
            <div className="flex space-x-2">
              <Link to="/">
                <Button 
                  variant={isActive("/") ? "default" : "ghost"}
                  size="lg"
                  className={`flex items-center space-x-2 font-medium transition-all duration-300 ${
                    isActive("/") 
                      ? "gradient-primary shadow-primary text-white" 
                      : "hover:bg-primary/10 hover:text-primary"
                  }`}
                >
                  <FileText className="w-5 h-5" />
                  <span>Upload</span>
                </Button>
              </Link>
              
              <Link to="/dashboard">
                <Button 
                  variant={isActive("/dashboard") ? "default" : "ghost"}
                  size="lg"
                  className={`flex items-center space-x-2 font-medium transition-all duration-300 ${
                    isActive("/dashboard") 
                      ? "gradient-primary shadow-primary text-white" 
                      : "hover:bg-primary/10 hover:text-primary"
                  }`}
                >
                  <LayoutDashboard className="w-5 h-5" />
                  <span>Dashboard</span>
                </Button>
              </Link>
              
              <Link to="/users">
                <Button 
                  variant={isActive("/users") ? "default" : "ghost"}
                  size="lg"
                  className={`flex items-center space-x-2 font-medium transition-all duration-300 ${
                    isActive("/users") 
                      ? "gradient-primary shadow-primary text-white" 
                      : "hover:bg-primary/10 hover:text-primary"
                  }`}
                >
                  <Users className="w-5 h-5" />
                  <span>Create User</span>
                </Button>
              </Link>

              <Link to="/users-list">
                <Button 
                  variant={isActive("/users-list") ? "default" : "ghost"}
                  size="lg"
                  className={`flex items-center space-x-2 font-medium transition-all duration-300 ${
                    isActive("/users-list") 
                      ? "gradient-primary shadow-primary text-white" 
                      : "hover:bg-primary/10 hover:text-primary"
                  }`}
                >
                  <UserCheck className="w-5 h-5" />
                  <span>Users List</span>
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;