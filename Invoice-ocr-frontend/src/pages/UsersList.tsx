import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Edit, Trash2, Search, UserPlus, Mail, Shield, Calendar, MoreVertical } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Navbar from "@/components/Navbar";

// Mock data - replace with real data from your backend
const mockUsers = [
  {
    id: 1,
    fullName: "John Doe",
    email: "john.doe@company.com",
    role: "Admin",
    status: "Active",
    avatar: "",
    createdAt: "2024-01-15",
    lastLogin: "2024-01-20 10:30:00"
  },
  {
    id: 2,
    fullName: "Jane Smith",
    email: "jane.smith@company.com",
    role: "User",
    status: "Active",
    avatar: "",
    createdAt: "2024-01-16",
    lastLogin: "2024-01-19 14:22:00"
  },
  {
    id: 3,
    fullName: "Mike Johnson",
    email: "mike.johnson@company.com",
    role: "User",
    status: "Inactive",
    avatar: "",
    createdAt: "2024-01-17",
    lastLogin: "2024-01-18 09:15:00"
  },
  {
    id: 4,
    fullName: "Sarah Wilson",
    email: "sarah.wilson@company.com",
    role: "Manager",
    status: "Active",
    avatar: "",
    createdAt: "2024-01-18",
    lastLogin: "2024-01-20 16:45:00"
  },
  {
    id: 5,
    fullName: "Alex Brown",
    email: "alex.brown@company.com",
    role: "User",
    status: "Active",
    avatar: "",
    createdAt: "2024-01-19",
    lastLogin: "Never"
  }
];

const UsersList = () => {
  const [users, setUsers] = useState(mockUsers);
  const [searchTerm, setSearchTerm] = useState("");
  const { toast } = useToast();

  const filteredUsers = users.filter(user =>
    user.fullName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.role.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleDelete = (id: number) => {
    setUsers(users.filter(user => user.id !== id));
    toast({
      title: "User deleted",
      description: "The user has been successfully removed from the system.",
    });
  };

  const handleEdit = (id: number) => {
    toast({
      title: "Edit user",
      description: "Edit user modal would open here in a real application.",
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Active":
        return "bg-success/10 text-success border-success/20";
      case "Inactive":
        return "bg-warning/10 text-warning border-warning/20";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case "Admin":
        return "bg-danger/10 text-danger border-danger/20";
      case "Manager":
        return "bg-primary/10 text-primary border-primary/20";
      case "User":
        return "bg-secondary/10 text-secondary border-secondary/20";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getInitials = (name: string) => {
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
      <Navbar />
      <div className="container mx-auto px-6 py-12">
        <div className="space-y-8">
          {/* Header Section */}
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 animate-fade-in">
            <div>
              <h1 className="text-4xl font-display font-bold bg-gradient-to-r from-primary via-primary-variant to-secondary bg-clip-text text-transparent">
                Users Management
              </h1>
              <p className="text-muted-foreground text-lg mt-2">
                Manage all system users and their permissions
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button className="gradient-primary shadow-primary hover:shadow-glow transition-all duration-300">
                <UserPlus className="w-5 h-5 mr-2" />
                Add New User
              </Button>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid md:grid-cols-4 gap-6 animate-slide-up">
            {[
              {
                title: "Total Users",
                value: users.length,
                icon: "👥",
                color: "from-primary to-primary-variant"
              },
              {
                title: "Active Users",
                value: users.filter(u => u.status === "Active").length,
                icon: "✅",
                color: "from-success to-green-400"
              },
              {
                title: "Inactive Users",
                value: users.filter(u => u.status === "Inactive").length,
                icon: "⏸️",
                color: "from-warning to-orange-400"
              },
              {
                title: "Admins",
                value: users.filter(u => u.role === "Admin").length,
                icon: "👑",
                color: "from-danger to-red-400"
              }
            ].map((stat, index) => (
              <Card key={index} className="hover-lift border-0 gradient-card">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">{stat.title}</p>
                      <p className="text-3xl font-bold mt-2">{stat.value}</p>
                    </div>
                    <div className={`p-4 rounded-xl bg-gradient-to-r ${stat.color} text-white text-2xl`}>
                      {stat.icon}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Users Table */}
          <div className="relative animate-scale-in">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-secondary/10 rounded-3xl blur-xl"></div>
            <Card className="relative border-0 gradient-card shadow-elegant">
              <CardHeader className="pb-8">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                  <CardTitle className="text-2xl font-display">
                    All Users ({filteredUsers.length})
                  </CardTitle>
                  <div className="relative max-w-md w-full lg:w-80">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
                    <Input
                      placeholder="Search users..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-12 h-12 border-2 border-primary/20 focus:border-primary/50 bg-background/50"
                    />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border/50 hover:bg-transparent">
                        <TableHead className="font-semibold text-foreground">User</TableHead>
                        <TableHead className="font-semibold text-foreground">Role</TableHead>
                        <TableHead className="font-semibold text-foreground">Status</TableHead>
                        <TableHead className="font-semibold text-foreground">Created</TableHead>
                        <TableHead className="font-semibold text-foreground">Last Login</TableHead>
                        <TableHead className="text-center font-semibold text-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredUsers.map((user) => (
                        <TableRow key={user.id} className="border-border/30 hover:bg-primary/5 transition-colors">
                          <TableCell>
                            <div className="flex items-center gap-4">
                              <Avatar className="h-12 w-12 border-2 border-primary/20">
                                <AvatarImage src={user.avatar} alt={user.fullName} />
                                <AvatarFallback className="gradient-primary text-white font-semibold">
                                  {getInitials(user.fullName)}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <p className="font-semibold text-foreground">{user.fullName}</p>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                  <Mail className="w-4 h-4" />
                                  {user.email}
                                </div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className={`${getRoleColor(user.role)} border font-medium`}>
                              <Shield className="w-3 h-3 mr-1" />
                              {user.role}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className={`${getStatusColor(user.status)} border font-medium`}>
                              {user.status}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <Calendar className="w-4 h-4" />
                              {user.createdAt}
                            </div>
                          </TableCell>
                          <TableCell className="font-mono text-sm">
                            {user.lastLogin}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center justify-center gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleEdit(user.id)}
                                className="h-10 w-10 p-0 hover:bg-primary/10 hover:text-primary"
                              >
                                <Edit className="w-4 h-4" />
                              </Button>
                              
                              <AlertDialog>
                                <AlertDialogTrigger asChild>
                                  <Button 
                                    variant="ghost" 
                                    size="sm"
                                    className="h-10 w-10 p-0 hover:bg-destructive/10 hover:text-destructive"
                                  >
                                    <Trash2 className="w-4 h-4" />
                                  </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent className="border-0 gradient-card shadow-elegant">
                                  <AlertDialogHeader>
                                    <AlertDialogTitle className="text-xl font-display">Delete User</AlertDialogTitle>
                                    <AlertDialogDescription className="text-base">
                                      Are you sure you want to delete <strong>{user.fullName}</strong>? 
                                      This action cannot be undone and will remove all user data.
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel className="border-border/50">Cancel</AlertDialogCancel>
                                    <AlertDialogAction
                                      onClick={() => handleDelete(user.id)}
                                      className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                    >
                                      Delete User
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  
                  {filteredUsers.length === 0 && (
                    <div className="text-center py-12">
                      <div className="text-6xl mb-4">👥</div>
                      <h3 className="text-xl font-semibold mb-2">No users found</h3>
                      <p className="text-muted-foreground">No users match your search criteria.</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UsersList;