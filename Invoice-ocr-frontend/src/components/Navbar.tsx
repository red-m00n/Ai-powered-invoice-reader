import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { FileText, LayoutDashboard, Users, UserCheck, LogOut, User } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    toast({
      title: "Logged out successfully",
      description: "You have been logged out of your account.",
    });
    navigate("/login");
  };

  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const isAdmin = user.role === "admin";

  return (
    <nav className="border-b bg-card/50 backdrop-blur-md border-border/50 sticky top-0 z-50">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-20">
          <div className="flex items-center space-x-12">
            <Link to="/upload" className="flex items-center space-x-3 group">
              <div className="p-2 rounded-xl gradient-primary shadow-primary group-hover:shadow-glow transition-all duration-300">
                <FileText className="w-7 h-7 text-white" />
              </div>
              <span className="font-display font-bold text-2xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                OCR Invoice
              </span>
            </Link>
            
            <div className="flex space-x-2">
              <Link to="/upload">
                <Button 
                  variant={isActive("/upload") ? "default" : "ghost"}
                  size="lg"
                  className={`flex items-center space-x-2 font-medium transition-all duration-300 ${
                    isActive("/upload") 
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
              
              {/* Admin-only features */}
              {isAdmin && (
                <>
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
                </>
              )}
            </div>
          </div>

          {/* User info and logout */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
              <User className="w-4 h-4" />
              <span>{user.full_name || "User"}</span>
              <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
                {user.role || "user"}
              </span>
            </div>
            <Button
              onClick={handleLogout}
              variant="ghost"
              size="sm"
              className="flex items-center space-x-2 text-muted-foreground hover:text-destructive"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;